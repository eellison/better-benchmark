"""
Standalone repro captured via capture_hook.
Label: DebertaV2ForMaskedLM
Pattern hash: 3241cf5c738d
Shape hash: 0dafcf13
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
    def forward(self, arg0_1: "f32[]", arg1_1: "f32[]", arg2_1: "i64[8, 512]", arg3_1: "bf16[8, 512, 128100]", arg4_1: "f32[4096, 1]", arg5_1: "f32[4096, 1]", arg6_1: "bf16[8, 512, 128100]", arg7_1: "bf16[128100, 1536]", arg8_1: "bf16[4096, 1536]", arg9_1: "f32[1536]", arg10_1: "bf16[4096, 1536]", arg11_1: "f32[8, 512, 1]", arg12_1: "f32[8, 512, 1]", arg13_1: "bf16[1536, 1536]", arg14_1: "bf16[4096, 1536]", arg15_1: "f32[1536]", arg16_1: "f32[8, 512, 1536]", arg17_1: "f32[8, 512, 1]", arg18_1: "b8[8, 512, 1536]", arg19_1: "bf16[1536, 6144]", arg20_1: "bf16[4096, 6144]", arg21_1: "bf16[4096, 6144]", arg22_1: "bf16[6144, 1536]", arg23_1: "bf16[4096, 1536]", arg24_1: "f32[1536]", arg25_1: "f32[8, 512, 1536]", arg26_1: "f32[8, 512, 1]", arg27_1: "b8[8, 512, 1536]", arg28_1: "bf16[1536, 1536]", arg29_1: "bf16[4096, 1536]", arg30_1: "bf16[192, 512, 512]", arg31_1: "bf16[192, 64, 512]", arg32_1: "b8[8, 24, 512, 512]", arg33_1: "bf16[8, 24, 512, 512]", arg34_1: "f32[8, 24, 512, 1]", arg35_1: "f32[8, 24, 512, 1]", arg36_1: "bf16[192, 64, 512]", arg37_1: "bf16[192, 512, 64]", arg38_1: "bf16[1536, 1536]", arg39_1: "bf16[4096, 1536]", arg40_1: "bf16[1536, 1536]", arg41_1: "bf16[1536, 1536]", arg42_1: "f32[1536]", arg43_1: "f32[8, 512, 1536]", arg44_1: "f32[8, 512, 1]", arg45_1: "b8[8, 512, 1536]", arg46_1: "bf16[1536, 6144]", arg47_1: "bf16[4096, 6144]", arg48_1: "bf16[4096, 6144]", arg49_1: "bf16[6144, 1536]", arg50_1: "bf16[4096, 1536]", arg51_1: "f32[1536]", arg52_1: "f32[8, 512, 1536]", arg53_1: "f32[8, 512, 1]", arg54_1: "b8[8, 512, 1536]", arg55_1: "bf16[1536, 1536]", arg56_1: "bf16[4096, 1536]", arg57_1: "bf16[192, 512, 512]", arg58_1: "bf16[192, 64, 512]", arg59_1: "b8[8, 24, 512, 512]", arg60_1: "bf16[8, 24, 512, 512]", arg61_1: "f32[8, 24, 512, 1]", arg62_1: "f32[8, 24, 512, 1]", arg63_1: "bf16[192, 64, 512]", arg64_1: "bf16[192, 512, 64]", arg65_1: "bf16[1536, 1536]", arg66_1: "bf16[4096, 1536]", arg67_1: "bf16[1536, 1536]", arg68_1: "bf16[1536, 1536]", arg69_1: "f32[1536]", arg70_1: "f32[8, 512, 1536]", arg71_1: "f32[8, 512, 1]", arg72_1: "b8[8, 512, 1536]", arg73_1: "bf16[1536, 6144]", arg74_1: "bf16[4096, 6144]", arg75_1: "bf16[4096, 6144]", arg76_1: "bf16[6144, 1536]", arg77_1: "bf16[4096, 1536]", arg78_1: "f32[1536]", arg79_1: "f32[8, 512, 1536]", arg80_1: "f32[8, 512, 1]", arg81_1: "b8[8, 512, 1536]", arg82_1: "bf16[1536, 1536]", arg83_1: "bf16[4096, 1536]", arg84_1: "bf16[192, 512, 512]", arg85_1: "bf16[192, 64, 512]", arg86_1: "b8[8, 24, 512, 512]", arg87_1: "bf16[8, 24, 512, 512]", arg88_1: "f32[8, 24, 512, 1]", arg89_1: "f32[8, 24, 512, 1]", arg90_1: "bf16[192, 64, 512]", arg91_1: "bf16[192, 512, 64]", arg92_1: "bf16[1536, 1536]", arg93_1: "bf16[4096, 1536]", arg94_1: "bf16[1536, 1536]", arg95_1: "bf16[1536, 1536]", arg96_1: "f32[1536]", arg97_1: "f32[8, 512, 1536]", arg98_1: "f32[8, 512, 1]", arg99_1: "b8[8, 512, 1536]", arg100_1: "bf16[1536, 6144]", arg101_1: "bf16[4096, 6144]", arg102_1: "bf16[4096, 6144]", arg103_1: "bf16[6144, 1536]", arg104_1: "bf16[4096, 1536]", arg105_1: "f32[1536]", arg106_1: "f32[8, 512, 1536]", arg107_1: "f32[8, 512, 1]", arg108_1: "b8[8, 512, 1536]", arg109_1: "bf16[1536, 1536]", arg110_1: "bf16[4096, 1536]", arg111_1: "bf16[192, 512, 512]", arg112_1: "bf16[192, 64, 512]", arg113_1: "b8[8, 24, 512, 512]", arg114_1: "bf16[8, 24, 512, 512]", arg115_1: "f32[8, 24, 512, 1]", arg116_1: "f32[8, 24, 512, 1]", arg117_1: "bf16[192, 64, 512]", arg118_1: "bf16[192, 512, 64]", arg119_1: "bf16[1536, 1536]", arg120_1: "bf16[4096, 1536]", arg121_1: "bf16[1536, 1536]", arg122_1: "bf16[1536, 1536]", arg123_1: "f32[1536]", arg124_1: "f32[8, 512, 1536]", arg125_1: "f32[8, 512, 1]", arg126_1: "b8[8, 512, 1536]", arg127_1: "bf16[1536, 6144]", arg128_1: "bf16[4096, 6144]", arg129_1: "bf16[4096, 6144]", arg130_1: "bf16[6144, 1536]", arg131_1: "bf16[4096, 1536]", arg132_1: "f32[1536]", arg133_1: "f32[8, 512, 1536]", arg134_1: "f32[8, 512, 1]", arg135_1: "b8[8, 512, 1536]", arg136_1: "bf16[1536, 1536]", arg137_1: "bf16[4096, 1536]", arg138_1: "bf16[192, 512, 512]", arg139_1: "bf16[192, 64, 512]", arg140_1: "b8[8, 24, 512, 512]", arg141_1: "bf16[8, 24, 512, 512]", arg142_1: "f32[8, 24, 512, 1]", arg143_1: "f32[8, 24, 512, 1]", arg144_1: "bf16[192, 64, 512]", arg145_1: "bf16[192, 512, 64]", arg146_1: "bf16[1536, 1536]", arg147_1: "bf16[4096, 1536]", arg148_1: "bf16[1536, 1536]", arg149_1: "bf16[1536, 1536]", arg150_1: "f32[1536]", arg151_1: "f32[8, 512, 1536]", arg152_1: "f32[8, 512, 1]", arg153_1: "b8[8, 512, 1536]", arg154_1: "bf16[1536, 6144]", arg155_1: "bf16[4096, 6144]", arg156_1: "bf16[4096, 6144]", arg157_1: "bf16[6144, 1536]", arg158_1: "bf16[4096, 1536]", arg159_1: "f32[1536]", arg160_1: "f32[8, 512, 1536]", arg161_1: "f32[8, 512, 1]", arg162_1: "b8[8, 512, 1536]", arg163_1: "bf16[1536, 1536]", arg164_1: "bf16[4096, 1536]", arg165_1: "bf16[192, 512, 512]", arg166_1: "bf16[192, 64, 512]", arg167_1: "b8[8, 24, 512, 512]", arg168_1: "bf16[8, 24, 512, 512]", arg169_1: "f32[8, 24, 512, 1]", arg170_1: "f32[8, 24, 512, 1]", arg171_1: "bf16[192, 64, 512]", arg172_1: "bf16[192, 512, 64]", arg173_1: "bf16[1536, 1536]", arg174_1: "bf16[4096, 1536]", arg175_1: "bf16[1536, 1536]", arg176_1: "bf16[1536, 1536]", arg177_1: "f32[1536]", arg178_1: "f32[8, 512, 1536]", arg179_1: "f32[8, 512, 1]", arg180_1: "b8[8, 512, 1536]", arg181_1: "bf16[1536, 6144]", arg182_1: "bf16[4096, 6144]", arg183_1: "bf16[4096, 6144]", arg184_1: "bf16[6144, 1536]", arg185_1: "bf16[4096, 1536]", arg186_1: "f32[1536]", arg187_1: "f32[8, 512, 1536]", arg188_1: "f32[8, 512, 1]", arg189_1: "b8[8, 512, 1536]", arg190_1: "bf16[1536, 1536]", arg191_1: "bf16[4096, 1536]", arg192_1: "bf16[192, 512, 512]", arg193_1: "bf16[192, 64, 512]", arg194_1: "b8[8, 24, 512, 512]", arg195_1: "bf16[8, 24, 512, 512]", arg196_1: "f32[8, 24, 512, 1]", arg197_1: "f32[8, 24, 512, 1]", arg198_1: "bf16[192, 64, 512]", arg199_1: "bf16[192, 512, 64]", arg200_1: "bf16[1536, 1536]", arg201_1: "bf16[4096, 1536]", arg202_1: "bf16[1536, 1536]", arg203_1: "bf16[1536, 1536]", arg204_1: "f32[1536]", arg205_1: "f32[8, 512, 1536]", arg206_1: "f32[8, 512, 1]", arg207_1: "b8[8, 512, 1536]", arg208_1: "bf16[1536, 6144]", arg209_1: "bf16[4096, 6144]", arg210_1: "bf16[4096, 6144]", arg211_1: "bf16[6144, 1536]", arg212_1: "bf16[4096, 1536]", arg213_1: "f32[1536]", arg214_1: "f32[8, 512, 1536]", arg215_1: "f32[8, 512, 1]", arg216_1: "b8[8, 512, 1536]", arg217_1: "bf16[1536, 1536]", arg218_1: "bf16[4096, 1536]", arg219_1: "bf16[192, 512, 512]", arg220_1: "bf16[192, 64, 512]", arg221_1: "b8[8, 24, 512, 512]", arg222_1: "bf16[8, 24, 512, 512]", arg223_1: "f32[8, 24, 512, 1]", arg224_1: "f32[8, 24, 512, 1]", arg225_1: "bf16[192, 64, 512]", arg226_1: "bf16[192, 512, 64]", arg227_1: "bf16[1536, 1536]", arg228_1: "bf16[4096, 1536]", arg229_1: "bf16[1536, 1536]", arg230_1: "bf16[1536, 1536]", arg231_1: "f32[1536]", arg232_1: "f32[8, 512, 1536]", arg233_1: "f32[8, 512, 1]", arg234_1: "b8[8, 512, 1536]", arg235_1: "bf16[1536, 6144]", arg236_1: "bf16[4096, 6144]", arg237_1: "bf16[4096, 6144]", arg238_1: "bf16[6144, 1536]", arg239_1: "bf16[4096, 1536]", arg240_1: "f32[1536]", arg241_1: "f32[8, 512, 1536]", arg242_1: "f32[8, 512, 1]", arg243_1: "b8[8, 512, 1536]", arg244_1: "bf16[1536, 1536]", arg245_1: "bf16[4096, 1536]", arg246_1: "bf16[192, 512, 512]", arg247_1: "bf16[192, 64, 512]", arg248_1: "b8[8, 24, 512, 512]", arg249_1: "bf16[8, 24, 512, 512]", arg250_1: "f32[8, 24, 512, 1]", arg251_1: "f32[8, 24, 512, 1]", arg252_1: "bf16[192, 64, 512]", arg253_1: "bf16[192, 512, 64]", arg254_1: "bf16[1536, 1536]", arg255_1: "bf16[4096, 1536]", arg256_1: "bf16[1536, 1536]", arg257_1: "bf16[1536, 1536]", arg258_1: "f32[1536]", arg259_1: "f32[8, 512, 1536]", arg260_1: "f32[8, 512, 1]", arg261_1: "b8[8, 512, 1536]", arg262_1: "bf16[1536, 6144]", arg263_1: "bf16[4096, 6144]", arg264_1: "bf16[4096, 6144]", arg265_1: "bf16[6144, 1536]", arg266_1: "bf16[4096, 1536]", arg267_1: "f32[1536]", arg268_1: "f32[8, 512, 1536]", arg269_1: "f32[8, 512, 1]", arg270_1: "b8[8, 512, 1536]", arg271_1: "bf16[1536, 1536]", arg272_1: "bf16[4096, 1536]", arg273_1: "bf16[192, 512, 512]", arg274_1: "bf16[192, 64, 512]", arg275_1: "b8[8, 24, 512, 512]", arg276_1: "bf16[8, 24, 512, 512]", arg277_1: "f32[8, 24, 512, 1]", arg278_1: "f32[8, 24, 512, 1]", arg279_1: "bf16[192, 64, 512]", arg280_1: "bf16[192, 512, 64]", arg281_1: "bf16[1536, 1536]", arg282_1: "bf16[4096, 1536]", arg283_1: "bf16[1536, 1536]", arg284_1: "bf16[1536, 1536]", arg285_1: "f32[1536]", arg286_1: "f32[8, 512, 1536]", arg287_1: "f32[8, 512, 1]", arg288_1: "b8[8, 512, 1536]", arg289_1: "bf16[1536, 6144]", arg290_1: "bf16[4096, 6144]", arg291_1: "bf16[4096, 6144]", arg292_1: "bf16[6144, 1536]", arg293_1: "bf16[4096, 1536]", arg294_1: "f32[1536]", arg295_1: "f32[8, 512, 1536]", arg296_1: "f32[8, 512, 1]", arg297_1: "b8[8, 512, 1536]", arg298_1: "bf16[1536, 1536]", arg299_1: "bf16[4096, 1536]", arg300_1: "bf16[192, 512, 512]", arg301_1: "bf16[192, 64, 512]", arg302_1: "b8[8, 24, 512, 512]", arg303_1: "bf16[8, 24, 512, 512]", arg304_1: "f32[8, 24, 512, 1]", arg305_1: "f32[8, 24, 512, 1]", arg306_1: "bf16[192, 64, 512]", arg307_1: "bf16[192, 512, 64]", arg308_1: "bf16[1536, 1536]", arg309_1: "bf16[4096, 1536]", arg310_1: "bf16[1536, 1536]", arg311_1: "bf16[1536, 1536]", arg312_1: "f32[1536]", arg313_1: "f32[8, 512, 1536]", arg314_1: "f32[8, 512, 1]", arg315_1: "b8[8, 512, 1536]", arg316_1: "bf16[1536, 6144]", arg317_1: "bf16[4096, 6144]", arg318_1: "bf16[4096, 6144]", arg319_1: "bf16[6144, 1536]", arg320_1: "bf16[4096, 1536]", arg321_1: "f32[1536]", arg322_1: "f32[8, 512, 1536]", arg323_1: "f32[8, 512, 1]", arg324_1: "b8[8, 512, 1536]", arg325_1: "bf16[1536, 1536]", arg326_1: "bf16[4096, 1536]", arg327_1: "bf16[192, 512, 512]", arg328_1: "bf16[192, 64, 512]", arg329_1: "b8[8, 24, 512, 512]", arg330_1: "bf16[8, 24, 512, 512]", arg331_1: "f32[8, 24, 512, 1]", arg332_1: "f32[8, 24, 512, 1]", arg333_1: "bf16[192, 64, 512]", arg334_1: "bf16[192, 512, 64]", arg335_1: "bf16[1536, 1536]", arg336_1: "bf16[4096, 1536]", arg337_1: "bf16[1536, 1536]", arg338_1: "bf16[1536, 1536]", arg339_1: "f32[1536]", arg340_1: "f32[8, 512, 1536]", arg341_1: "f32[8, 512, 1]", arg342_1: "b8[8, 512, 1536]", arg343_1: "bf16[1536, 6144]", arg344_1: "bf16[4096, 6144]", arg345_1: "bf16[4096, 6144]", arg346_1: "bf16[6144, 1536]", arg347_1: "bf16[4096, 1536]", arg348_1: "f32[1536]", arg349_1: "f32[8, 512, 1536]", arg350_1: "f32[8, 512, 1]", arg351_1: "b8[8, 512, 1536]", arg352_1: "bf16[1536, 1536]", arg353_1: "bf16[4096, 1536]", arg354_1: "bf16[192, 512, 512]", arg355_1: "bf16[192, 64, 512]", arg356_1: "b8[8, 24, 512, 512]", arg357_1: "bf16[8, 24, 512, 512]", arg358_1: "f32[8, 24, 512, 1]", arg359_1: "f32[8, 24, 512, 1]", arg360_1: "bf16[192, 64, 512]", arg361_1: "bf16[192, 512, 64]", arg362_1: "bf16[1536, 1536]", arg363_1: "bf16[4096, 1536]", arg364_1: "bf16[1536, 1536]", arg365_1: "bf16[1536, 1536]", arg366_1: "f32[1536]", arg367_1: "f32[8, 512, 1536]", arg368_1: "f32[8, 512, 1]", arg369_1: "b8[8, 512, 1536]", arg370_1: "bf16[1536, 6144]", arg371_1: "bf16[4096, 6144]", arg372_1: "bf16[4096, 6144]", arg373_1: "bf16[6144, 1536]", arg374_1: "bf16[4096, 1536]", arg375_1: "f32[1536]", arg376_1: "f32[8, 512, 1536]", arg377_1: "f32[8, 512, 1]", arg378_1: "b8[8, 512, 1536]", arg379_1: "bf16[1536, 1536]", arg380_1: "bf16[4096, 1536]", arg381_1: "bf16[192, 512, 512]", arg382_1: "bf16[192, 64, 512]", arg383_1: "b8[8, 24, 512, 512]", arg384_1: "bf16[8, 24, 512, 512]", arg385_1: "f32[8, 24, 512, 1]", arg386_1: "f32[8, 24, 512, 1]", arg387_1: "bf16[192, 64, 512]", arg388_1: "bf16[192, 512, 64]", arg389_1: "bf16[1536, 1536]", arg390_1: "bf16[4096, 1536]", arg391_1: "bf16[1536, 1536]", arg392_1: "bf16[1536, 1536]", arg393_1: "f32[1536]", arg394_1: "f32[8, 512, 1536]", arg395_1: "f32[8, 512, 1]", arg396_1: "b8[8, 512, 1536]", arg397_1: "bf16[1536, 6144]", arg398_1: "bf16[4096, 6144]", arg399_1: "bf16[4096, 6144]", arg400_1: "bf16[6144, 1536]", arg401_1: "bf16[4096, 1536]", arg402_1: "f32[1536]", arg403_1: "f32[8, 512, 1536]", arg404_1: "f32[8, 512, 1]", arg405_1: "b8[8, 512, 1536]", arg406_1: "bf16[1536, 1536]", arg407_1: "bf16[4096, 1536]", arg408_1: "bf16[192, 512, 512]", arg409_1: "bf16[192, 64, 512]", arg410_1: "b8[8, 24, 512, 512]", arg411_1: "bf16[8, 24, 512, 512]", arg412_1: "f32[8, 24, 512, 1]", arg413_1: "f32[8, 24, 512, 1]", arg414_1: "bf16[192, 64, 512]", arg415_1: "bf16[192, 512, 64]", arg416_1: "bf16[1536, 1536]", arg417_1: "bf16[4096, 1536]", arg418_1: "bf16[1536, 1536]", arg419_1: "bf16[1536, 1536]", arg420_1: "f32[1536]", arg421_1: "f32[8, 512, 1536]", arg422_1: "f32[8, 512, 1]", arg423_1: "b8[8, 512, 1536]", arg424_1: "bf16[1536, 6144]", arg425_1: "bf16[4096, 6144]", arg426_1: "bf16[4096, 6144]", arg427_1: "bf16[6144, 1536]", arg428_1: "bf16[4096, 1536]", arg429_1: "f32[1536]", arg430_1: "f32[8, 512, 1536]", arg431_1: "f32[8, 512, 1]", arg432_1: "b8[8, 512, 1536]", arg433_1: "bf16[1536, 1536]", arg434_1: "bf16[4096, 1536]", arg435_1: "bf16[192, 512, 512]", arg436_1: "bf16[192, 64, 512]", arg437_1: "b8[8, 24, 512, 512]", arg438_1: "bf16[8, 24, 512, 512]", arg439_1: "f32[8, 24, 512, 1]", arg440_1: "f32[8, 24, 512, 1]", arg441_1: "bf16[192, 64, 512]", arg442_1: "bf16[192, 512, 64]", arg443_1: "bf16[1536, 1536]", arg444_1: "bf16[4096, 1536]", arg445_1: "bf16[1536, 1536]", arg446_1: "bf16[1536, 1536]", arg447_1: "f32[1536]", arg448_1: "f32[8, 512, 1536]", arg449_1: "f32[8, 512, 1]", arg450_1: "b8[8, 512, 1536]", arg451_1: "bf16[1536, 6144]", arg452_1: "bf16[4096, 6144]", arg453_1: "bf16[4096, 6144]", arg454_1: "bf16[6144, 1536]", arg455_1: "bf16[4096, 1536]", arg456_1: "f32[1536]", arg457_1: "f32[8, 512, 1536]", arg458_1: "f32[8, 512, 1]", arg459_1: "b8[8, 512, 1536]", arg460_1: "bf16[1536, 1536]", arg461_1: "bf16[4096, 1536]", arg462_1: "bf16[192, 512, 512]", arg463_1: "bf16[192, 64, 512]", arg464_1: "b8[8, 24, 512, 512]", arg465_1: "bf16[8, 24, 512, 512]", arg466_1: "f32[8, 24, 512, 1]", arg467_1: "f32[8, 24, 512, 1]", arg468_1: "bf16[192, 64, 512]", arg469_1: "bf16[192, 512, 64]", arg470_1: "bf16[1536, 1536]", arg471_1: "bf16[4096, 1536]", arg472_1: "bf16[1536, 1536]", arg473_1: "bf16[1536, 1536]", arg474_1: "f32[1536]", arg475_1: "f32[8, 512, 1536]", arg476_1: "f32[8, 512, 1]", arg477_1: "b8[8, 512, 1536]", arg478_1: "bf16[1536, 6144]", arg479_1: "bf16[4096, 6144]", arg480_1: "bf16[4096, 6144]", arg481_1: "bf16[6144, 1536]", arg482_1: "bf16[4096, 1536]", arg483_1: "f32[1536]", arg484_1: "f32[8, 512, 1536]", arg485_1: "f32[8, 512, 1]", arg486_1: "b8[8, 512, 1536]", arg487_1: "bf16[1536, 1536]", arg488_1: "bf16[4096, 1536]", arg489_1: "bf16[192, 512, 512]", arg490_1: "bf16[192, 64, 512]", arg491_1: "b8[8, 24, 512, 512]", arg492_1: "bf16[8, 24, 512, 512]", arg493_1: "f32[8, 24, 512, 1]", arg494_1: "f32[8, 24, 512, 1]", arg495_1: "bf16[192, 64, 512]", arg496_1: "bf16[192, 512, 64]", arg497_1: "bf16[1536, 1536]", arg498_1: "bf16[4096, 1536]", arg499_1: "bf16[1536, 1536]", arg500_1: "bf16[1536, 1536]", arg501_1: "f32[1536]", arg502_1: "f32[8, 512, 1536]", arg503_1: "f32[8, 512, 1]", arg504_1: "b8[8, 512, 1536]", arg505_1: "bf16[1536, 6144]", arg506_1: "bf16[4096, 6144]", arg507_1: "bf16[4096, 6144]", arg508_1: "bf16[6144, 1536]", arg509_1: "bf16[4096, 1536]", arg510_1: "f32[1536]", arg511_1: "f32[8, 512, 1536]", arg512_1: "f32[8, 512, 1]", arg513_1: "b8[8, 512, 1536]", arg514_1: "bf16[1536, 1536]", arg515_1: "bf16[4096, 1536]", arg516_1: "bf16[192, 512, 512]", arg517_1: "bf16[192, 64, 512]", arg518_1: "b8[8, 24, 512, 512]", arg519_1: "bf16[8, 24, 512, 512]", arg520_1: "f32[8, 24, 512, 1]", arg521_1: "f32[8, 24, 512, 1]", arg522_1: "bf16[192, 64, 512]", arg523_1: "bf16[192, 512, 64]", arg524_1: "bf16[1536, 1536]", arg525_1: "bf16[4096, 1536]", arg526_1: "bf16[1536, 1536]", arg527_1: "bf16[1536, 1536]", arg528_1: "f32[1536]", arg529_1: "f32[8, 512, 1536]", arg530_1: "f32[8, 512, 1]", arg531_1: "b8[8, 512, 1536]", arg532_1: "bf16[1536, 6144]", arg533_1: "bf16[4096, 6144]", arg534_1: "bf16[4096, 6144]", arg535_1: "bf16[6144, 1536]", arg536_1: "bf16[4096, 1536]", arg537_1: "f32[1536]", arg538_1: "f32[8, 512, 1536]", arg539_1: "f32[8, 512, 1]", arg540_1: "b8[8, 512, 1536]", arg541_1: "bf16[1536, 1536]", arg542_1: "bf16[4096, 1536]", arg543_1: "bf16[192, 512, 512]", arg544_1: "bf16[192, 64, 512]", arg545_1: "b8[8, 24, 512, 512]", arg546_1: "bf16[8, 24, 512, 512]", arg547_1: "f32[8, 24, 512, 1]", arg548_1: "f32[8, 24, 512, 1]", arg549_1: "bf16[192, 64, 512]", arg550_1: "bf16[192, 512, 64]", arg551_1: "bf16[1536, 1536]", arg552_1: "bf16[4096, 1536]", arg553_1: "bf16[1536, 1536]", arg554_1: "bf16[1536, 1536]", arg555_1: "f32[1536]", arg556_1: "f32[8, 512, 1536]", arg557_1: "f32[8, 512, 1]", arg558_1: "b8[8, 512, 1536]", arg559_1: "bf16[1536, 6144]", arg560_1: "bf16[4096, 6144]", arg561_1: "bf16[4096, 6144]", arg562_1: "bf16[6144, 1536]", arg563_1: "bf16[4096, 1536]", arg564_1: "f32[1536]", arg565_1: "f32[8, 512, 1536]", arg566_1: "f32[8, 512, 1]", arg567_1: "b8[8, 512, 1536]", arg568_1: "bf16[1536, 1536]", arg569_1: "bf16[4096, 1536]", arg570_1: "bf16[192, 512, 512]", arg571_1: "bf16[192, 64, 512]", arg572_1: "b8[8, 24, 512, 512]", arg573_1: "bf16[8, 24, 512, 512]", arg574_1: "f32[8, 24, 512, 1]", arg575_1: "f32[8, 24, 512, 1]", arg576_1: "bf16[192, 64, 512]", arg577_1: "bf16[192, 512, 64]", arg578_1: "bf16[1536, 1536]", arg579_1: "bf16[4096, 1536]", arg580_1: "bf16[1536, 1536]", arg581_1: "bf16[1536, 1536]", arg582_1: "f32[1536]", arg583_1: "f32[8, 512, 1536]", arg584_1: "f32[8, 512, 1]", arg585_1: "b8[8, 512, 1536]", arg586_1: "bf16[1536, 6144]", arg587_1: "bf16[4096, 6144]", arg588_1: "bf16[4096, 6144]", arg589_1: "bf16[6144, 1536]", arg590_1: "bf16[4096, 1536]", arg591_1: "f32[1536]", arg592_1: "f32[8, 512, 1536]", arg593_1: "f32[8, 512, 1]", arg594_1: "b8[8, 512, 1536]", arg595_1: "bf16[1536, 1536]", arg596_1: "bf16[4096, 1536]", arg597_1: "bf16[192, 512, 512]", arg598_1: "bf16[192, 64, 512]", arg599_1: "b8[8, 24, 512, 512]", arg600_1: "bf16[8, 24, 512, 512]", arg601_1: "f32[8, 24, 512, 1]", arg602_1: "f32[8, 24, 512, 1]", arg603_1: "bf16[192, 64, 512]", arg604_1: "bf16[192, 512, 64]", arg605_1: "bf16[1536, 1536]", arg606_1: "bf16[4096, 1536]", arg607_1: "bf16[1536, 1536]", arg608_1: "bf16[1536, 1536]", arg609_1: "f32[1536]", arg610_1: "f32[8, 512, 1536]", arg611_1: "f32[8, 512, 1]", arg612_1: "b8[8, 512, 1536]", arg613_1: "bf16[1536, 6144]", arg614_1: "bf16[4096, 6144]", arg615_1: "bf16[4096, 6144]", arg616_1: "bf16[6144, 1536]", arg617_1: "bf16[4096, 1536]", arg618_1: "f32[1536]", arg619_1: "f32[8, 512, 1536]", arg620_1: "f32[8, 512, 1]", arg621_1: "b8[8, 512, 1536]", arg622_1: "bf16[1536, 1536]", arg623_1: "bf16[4096, 1536]", arg624_1: "bf16[192, 512, 512]", arg625_1: "bf16[192, 64, 512]", arg626_1: "b8[8, 24, 512, 512]", arg627_1: "bf16[8, 24, 512, 512]", arg628_1: "f32[8, 24, 512, 1]", arg629_1: "f32[8, 24, 512, 1]", arg630_1: "bf16[192, 64, 512]", arg631_1: "bf16[192, 512, 64]", arg632_1: "bf16[1536, 1536]", arg633_1: "bf16[4096, 1536]", arg634_1: "bf16[1536, 1536]", arg635_1: "bf16[1536, 1536]", arg636_1: "f32[1536]", arg637_1: "f32[8, 512, 1536]", arg638_1: "f32[8, 512, 1]", arg639_1: "b8[8, 512, 1536]", arg640_1: "bf16[1536, 6144]", arg641_1: "bf16[4096, 6144]", arg642_1: "bf16[4096, 6144]", arg643_1: "bf16[6144, 1536]", arg644_1: "bf16[4096, 1536]", arg645_1: "f32[1536]", arg646_1: "f32[8, 512, 1536]", arg647_1: "f32[8, 512, 1]", arg648_1: "b8[8, 512, 1536]", arg649_1: "bf16[1536, 1536]", arg650_1: "bf16[4096, 1536]", arg651_1: "bf16[192, 512, 512]", arg652_1: "bf16[192, 64, 512]", arg653_1: "b8[8, 24, 512, 512]", arg654_1: "bf16[192, 512, 512]", arg655_1: "f32[8, 24, 512, 1]", arg656_1: "f32[8, 24, 512, 1]", arg657_1: "bf16[192, 64, 512]", arg658_1: "bf16[192, 512, 64]", arg659_1: "bf16[1536, 1536]", arg660_1: "bf16[4096, 1536]", arg661_1: "bf16[1536, 1536]", arg662_1: "bf16[1536, 1536]", arg663_1: "b8[8, 512, 1536]", arg664_1: "f32[1536]", arg665_1: "f32[8, 512, 1536]", arg666_1: "f32[1, 512, 1536]", arg667_1: "f32[8, 512, 1]", arg668_1: "f32[8, 512, 1]", arg669_1: "i64[1, 512]", arg670_1: "i64[8, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31, _shape_param_32, _shape_param_33, _shape_param_34, _shape_param_35, _shape_param_36, _shape_param_37, _shape_param_38, _shape_param_39, _shape_param_40, _shape_param_41, _shape_param_42, _shape_param_43, _shape_param_44, _shape_param_45, _shape_param_46, _shape_param_47, _shape_param_48, _shape_param_49, _shape_param_50, _shape_param_51, _shape_param_52, _shape_param_53, _shape_param_54, _shape_param_55, _shape_param_56, _shape_param_57, _shape_param_58, _shape_param_59, _shape_param_60, _shape_param_61, _shape_param_62, _shape_param_63, _shape_param_64, _shape_param_65, _shape_param_66, _shape_param_67, _shape_param_68, _shape_param_69, _shape_param_70, _shape_param_71, _shape_param_72, _shape_param_73, _shape_param_74, _shape_param_75, _shape_param_76, _shape_param_77, _shape_param_78, _shape_param_79, _shape_param_80, _shape_param_81, _shape_param_82, _shape_param_83, _shape_param_84, _shape_param_85, _shape_param_86, _shape_param_87, _shape_param_88, _shape_param_89, _shape_param_90, _shape_param_91, _shape_param_92, _shape_param_93, _shape_param_94, _shape_param_95, _shape_param_96, _shape_param_97, _shape_param_98, _shape_param_99, _shape_param_100, _shape_param_101, _shape_param_102, _shape_param_103, _shape_param_104, _shape_param_105, _shape_param_106, _shape_param_107, _shape_param_108, _shape_param_109, _shape_param_110, _shape_param_111, _shape_param_112, _shape_param_113, _shape_param_114, _shape_param_115, _shape_param_116, _shape_param_117, _shape_param_118, _shape_param_119, _shape_param_120, _shape_param_121, _shape_param_122, _shape_param_123, _shape_param_124, _shape_param_125, _shape_param_126, _shape_param_127, _shape_param_128, _shape_param_129, _shape_param_130, _shape_param_131, _shape_param_132, _shape_param_133, _shape_param_134, _shape_param_135, _shape_param_136, _shape_param_137, _shape_param_138, _shape_param_139, _shape_param_140, _shape_param_141, _shape_param_142, _shape_param_143, _shape_param_144, _shape_param_145, _shape_param_146, _shape_param_147, _shape_param_148, _shape_param_149, _shape_param_150, _shape_param_151, _shape_param_152, _shape_param_153, _shape_param_154, _shape_param_155, _shape_param_156, _shape_param_157, _shape_param_158, _shape_param_159, _shape_param_160, _shape_param_161, _shape_param_162, _shape_param_163, _shape_param_164, _shape_param_165, _shape_param_166, _shape_param_167, _shape_param_168, _shape_param_169, _shape_param_170, _shape_param_171, _shape_param_172, _shape_param_173, _shape_param_174, _shape_param_175, _shape_param_176, _shape_param_177, _shape_param_178, _shape_param_179, _shape_param_180, _shape_param_181, _shape_param_182, _shape_param_183, _shape_param_184, _shape_param_185, _shape_param_186, _shape_param_187, _shape_param_188, _shape_param_189, _shape_param_190, _shape_param_191, _shape_param_192, _shape_param_193, _shape_param_194, _shape_param_195, _shape_param_196, _shape_param_197, _shape_param_198, _shape_param_199, _shape_param_200, _shape_param_201, _shape_param_202, _shape_param_203, _shape_param_204, _shape_param_205, _shape_param_206, _shape_param_207, _shape_param_208, _shape_param_209, _shape_param_210, _shape_param_211, _shape_param_212, _shape_param_213, _shape_param_214, _shape_param_215, _shape_param_216, _shape_param_217, _shape_param_218, _shape_param_219, _shape_param_220, _shape_param_221, _shape_param_222, _shape_param_223, _shape_param_224, _shape_param_225, _shape_param_226, _shape_param_227, _shape_param_228, _shape_param_229, _shape_param_230, _shape_param_231, _shape_param_232, _shape_param_233, _shape_param_234, _shape_param_235, _shape_param_236, _shape_param_237, _shape_param_238, _shape_param_239, _shape_param_240, _shape_param_241, _shape_param_242, _shape_param_243, _shape_param_244, _shape_param_245, _shape_param_246, _shape_param_247, _shape_param_248, _shape_param_249, _shape_param_250, _shape_param_251, _shape_param_252, _shape_param_253, _shape_param_254, _shape_param_255, _shape_param_256, _shape_param_257, _shape_param_258, _shape_param_259, _shape_param_260, _shape_param_261, _shape_param_262, _shape_param_263, _shape_param_264, _shape_param_265, _shape_param_266, _shape_param_267, _shape_param_268, _shape_param_269, _shape_param_270, _shape_param_271, _shape_param_272, _shape_param_273, _shape_param_274, _shape_param_275, _shape_param_276, _shape_param_277, _shape_param_278, _shape_param_279, _shape_param_280, _shape_param_281, _shape_param_282, _shape_param_283, _shape_param_284, _shape_param_285, _shape_param_286, _shape_param_287, _shape_param_288, _shape_param_289, _shape_param_290, _shape_param_291, _shape_param_292, _shape_param_293, _shape_param_294, _shape_param_295, _shape_param_296, _shape_param_297, _shape_param_298, _shape_param_299, _shape_param_300, _shape_param_301, _shape_param_302, _shape_param_303, _shape_param_304, _shape_param_305, _shape_param_306, _shape_param_307, _shape_param_308, _shape_param_309, _shape_param_310, _shape_param_311, _shape_param_312, _shape_param_313, _shape_param_314, _shape_param_315, _shape_param_316, _shape_param_317, _shape_param_318, _shape_param_319, _shape_param_320, _shape_param_321, _shape_param_322, _shape_param_323, _shape_param_324, _shape_param_325, _shape_param_326, _shape_param_327, _shape_param_328, _shape_param_329, _shape_param_330, _shape_param_331, _shape_param_332, _shape_param_333, _shape_param_334, _shape_param_335, _shape_param_336, _shape_param_337, _shape_param_338, _shape_param_339, _shape_param_340, _shape_param_341, _shape_param_342, _shape_param_343, _shape_param_344, _shape_param_345, _shape_param_346, _shape_param_347, _shape_param_348, _shape_param_349, _shape_param_350, _shape_param_351, _shape_param_352, _shape_param_353, _shape_param_354, _shape_param_355, _shape_param_356, _shape_param_357, _shape_param_358, _shape_param_359, _shape_param_360, _shape_param_361, _shape_param_362, _shape_param_363, _shape_param_364, _shape_param_365, _shape_param_366, _shape_param_367, _shape_param_368, _shape_param_369, _shape_param_370, _shape_param_371, _shape_param_372, _shape_param_373, _shape_param_374, _shape_param_375, _shape_param_376, _shape_param_377, _shape_param_378, _shape_param_379, _shape_param_380, _shape_param_381, _shape_param_382, _shape_param_383, _shape_param_384, _shape_param_385, _shape_param_386, _shape_param_387, _shape_param_388, _shape_param_389, _shape_param_390, _shape_param_391, _shape_param_392, _shape_param_393, _shape_param_394, _shape_param_395, _shape_param_396, _shape_param_397, _shape_param_398, _shape_param_399, _shape_param_400, _shape_param_401, _shape_param_402, _shape_param_403, _shape_param_404, _shape_param_405, _shape_param_406, _shape_param_407, _shape_param_408, _shape_param_409, _shape_param_410, _shape_param_411, _shape_param_412, _shape_param_413, _shape_param_414, _shape_param_415, _shape_param_416, _shape_param_417, _shape_param_418, _shape_param_419, _shape_param_420, _shape_param_421, _shape_param_422, _shape_param_423, _shape_param_424, _shape_param_425, _shape_param_426, _shape_param_427, _shape_param_428, _shape_param_429, _shape_param_430, _shape_param_431, _shape_param_432, _shape_param_433, _shape_param_434, _shape_param_435, _shape_param_436, _shape_param_437, _shape_param_438, _shape_param_439, _shape_param_440, _shape_param_441, _shape_param_442, _shape_param_443, _shape_param_444, _shape_param_445, _shape_param_446, _shape_param_447, _shape_param_448, _shape_param_449, _shape_param_450, _shape_param_451, _shape_param_452, _shape_param_453, _shape_param_454, _shape_param_455, _shape_param_456, _shape_param_457, _shape_param_458, _shape_param_459, _shape_param_460, _shape_param_461, _shape_param_462, _shape_param_463, _shape_param_464, _shape_param_465, _shape_param_466, _shape_param_467, _shape_param_468, _shape_param_469, _shape_param_470, _shape_param_471, _shape_param_472, _shape_param_473, _shape_param_474, _shape_param_475, _shape_param_476, _shape_param_477, _shape_param_478, _shape_param_479, _shape_param_480, _shape_param_481, _shape_param_482, _shape_param_483, _shape_param_484, _shape_param_485, _shape_param_486, _shape_param_487, _shape_param_488, _shape_param_489, _shape_param_490, _shape_param_491, _shape_param_492, _shape_param_493, _shape_param_494, _shape_param_495, _shape_param_496, _shape_param_497, _shape_param_498, _shape_param_499, _shape_param_500, _shape_param_501, _shape_param_502, _shape_param_503, _shape_param_504, _shape_param_505, _shape_param_506, _shape_param_507, _shape_param_508, _shape_param_509, _shape_param_510, _shape_param_511, _shape_param_512, _shape_param_513, _shape_param_514, _shape_param_515, _shape_param_516, _shape_param_517, _shape_param_518, _shape_param_519, _shape_param_520, _shape_param_521, _shape_param_522, _shape_param_523, _shape_param_524, _shape_param_525, _shape_param_526, _shape_param_527, _shape_param_528, _shape_param_529, _shape_param_530, _shape_param_531, _shape_param_532, _shape_param_533, _shape_param_534, _shape_param_535, _shape_param_536, _shape_param_537, _shape_param_538, _shape_param_539, _shape_param_540, _shape_param_541, _shape_param_542, _shape_param_543, _shape_param_544, _shape_param_545, _shape_param_546, _shape_param_547, _shape_param_548, _shape_param_549, _shape_param_550, _shape_param_551, _shape_param_552, _shape_param_553, _shape_param_554, _shape_param_555, _shape_param_556, _shape_param_557, _shape_param_558, _shape_param_559, _shape_param_560, _shape_param_561, _shape_param_562, _shape_param_563, _shape_param_564, _shape_param_565, _shape_param_566, _shape_param_567, _shape_param_568, _shape_param_569, _shape_param_570, _shape_param_571, _shape_param_572, _shape_param_573, _shape_param_574, _shape_param_575, _shape_param_576, _shape_param_577, _shape_param_578, _shape_param_579, _shape_param_580, _shape_param_581, _shape_param_582, _shape_param_583, _shape_param_584, _shape_param_585, _shape_param_586, _shape_param_587, _shape_param_588, _shape_param_589, _shape_param_590, _shape_param_591, _shape_param_592, _shape_param_593, _shape_param_594, _shape_param_595, _shape_param_596, _shape_param_597, _shape_param_598, _shape_param_599, _shape_param_600, _shape_param_601, _shape_param_602, _shape_param_603, _shape_param_604, _shape_param_605, _shape_param_606, _shape_param_607, _shape_param_608, _shape_param_609, _shape_param_610, _shape_param_611, _shape_param_612, _shape_param_613, _shape_param_614, _shape_param_615, _shape_param_616, _shape_param_617, _shape_param_618, _shape_param_619, _shape_param_620, _shape_param_621, _shape_param_622, _shape_param_623, _shape_param_624, _shape_param_625, _shape_param_626, _shape_param_627, _shape_param_628, _shape_param_629, _shape_param_630, _shape_param_631, _shape_param_632, _shape_param_633, _shape_param_634, _shape_param_635, _shape_param_636, _shape_param_637, _shape_param_638, _shape_param_639, _shape_param_640, _shape_param_641, _shape_param_642, _shape_param_643, _shape_param_644, _shape_param_645, _shape_param_646, _shape_param_647, _shape_param_648, _shape_param_649, _shape_param_650, _shape_param_651, _shape_param_652, _shape_param_653, _shape_param_654, _shape_param_655, _shape_param_656, _shape_param_657, _shape_param_658, _shape_param_659, _shape_param_660, _shape_param_661, _shape_param_662, _shape_param_663, _shape_param_664, _shape_param_665, _shape_param_666, _shape_param_667, _shape_param_668, _shape_param_669, _shape_param_670, _shape_param_671, _shape_param_672, _shape_param_673, _shape_param_674, _shape_param_675, _shape_param_676, _shape_param_677, _shape_param_678, _shape_param_679, _shape_param_680, _shape_param_681, _shape_param_682, _shape_param_683, _shape_param_684, _shape_param_685, _shape_param_686, _shape_param_687, _shape_param_688, _shape_param_689, _shape_param_690, _shape_param_691, _shape_param_692, _shape_param_693, _shape_param_694, _shape_param_695, _shape_param_696, _shape_param_697, _shape_param_698, _shape_param_699, _shape_param_700, _shape_param_701, _shape_param_702, _shape_param_703, _shape_param_704, _shape_param_705, _shape_param_706, _shape_param_707, _shape_param_708, _shape_param_709, _shape_param_710, _shape_param_711, _shape_param_712, _shape_param_713):
        # No stacktrace found for following nodes
        div: "f32[]" = torch.ops.aten.div.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "i64[4096]" = torch.ops.aten.view.default(arg2_1, [-1]);  arg2_1 = None
        unsqueeze: "i64[4096, 1]" = torch.ops.aten.unsqueeze.default(view, 1);  view = None
        ne: "b8[4096, 1]" = torch.ops.aten.ne.Scalar(unsqueeze, -100)
        full: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "i64[4096, 1]" = torch.ops.aten.where.self(ne, unsqueeze, full);  unsqueeze = full = None
        iota: "i64[128100]" = torch.ops.prims.iota.default(128100, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view_1: "i64[1, 128100]" = torch.ops.aten.view.default(iota, _shape_param_0);  iota = _shape_param_0 = None
        expand: "i64[4096, 128100]" = torch.ops.aten.expand.default(where, _shape_param_1);  where = _shape_param_1 = None
        eq: "b8[4096, 128100]" = torch.ops.aten.eq.Tensor(expand, view_1);  expand = view_1 = None
        scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        where_1: "f32[4096, 128100]" = torch.ops.aten.where.self(eq, scalar_tensor_1, scalar_tensor);  eq = scalar_tensor_1 = scalar_tensor = None
        full_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "f32[4096, 1]" = torch.ops.aten.where.self(ne, div, full_1);  ne = div = full_1 = None
        mul: "f32[4096, 128100]" = torch.ops.aten.mul.Tensor(where_1, where_2);  where_1 = where_2 = None
        convert_element_type: "bf16[4096, 128100]" = torch.ops.prims.convert_element_type.default(mul, torch.bfloat16);  mul = None
        convert_element_type_1: "f32[4096, 128100]" = torch.ops.prims.convert_element_type.default(convert_element_type, torch.float32);  convert_element_type = None
        view_2: "bf16[4096, 128100]" = torch.ops.aten.view.default(arg3_1, _shape_param_2);  arg3_1 = _shape_param_2 = None
        convert_element_type_2: "f32[4096, 128100]" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        sub: "f32[4096, 128100]" = torch.ops.aten.sub.Tensor(convert_element_type_2, arg4_1);  convert_element_type_2 = arg4_1 = None
        sub_1: "f32[4096, 128100]" = torch.ops.aten.sub.Tensor(sub, arg5_1);  sub = arg5_1 = None
        convert_element_type_3: "bf16[4096, 128100]" = torch.ops.prims.convert_element_type.default(sub_1, torch.bfloat16);  sub_1 = None
        convert_element_type_4: "f32[4096, 128100]" = torch.ops.prims.convert_element_type.default(convert_element_type_3, torch.float32);  convert_element_type_3 = None
        exp: "f32[4096, 128100]" = torch.ops.aten.exp.default(convert_element_type_4);  convert_element_type_4 = None
        sum_1: "f32[4096, 1]" = torch.ops.aten.sum.dim_IntList(convert_element_type_1, [1], True)
        mul_1: "f32[4096, 128100]" = torch.ops.aten.mul.Tensor(exp, sum_1);  exp = sum_1 = None
        sub_2: "f32[4096, 128100]" = torch.ops.aten.sub.Tensor(convert_element_type_1, mul_1);  convert_element_type_1 = mul_1 = None
        convert_element_type_5: "bf16[4096, 128100]" = torch.ops.prims.convert_element_type.default(sub_2, torch.bfloat16);  sub_2 = None
        view_3: "bf16[8, 512, 128100]" = torch.ops.aten.view.default(convert_element_type_5, _shape_param_3);  convert_element_type_5 = _shape_param_3 = None
        add: "bf16[8, 512, 128100]" = torch.ops.aten.add.Tensor(arg6_1, view_3);  arg6_1 = view_3 = None
        view_4: "bf16[4096, 128100]" = torch.ops.aten.view.default(add, _shape_param_4);  add = _shape_param_4 = None
        constant_pad_nd: "bf16[4096, 128104]" = torch.ops.aten.constant_pad_nd.default(view_4, _shape_param_5);  _shape_param_5 = None
        constant_pad_nd_1: "bf16[128104, 1536]" = torch.ops.aten.constant_pad_nd.default(arg7_1, _shape_param_6);  arg7_1 = _shape_param_6 = None
        mm: "bf16[4096, 1536]" = torch.ops.aten.mm.default(constant_pad_nd, constant_pad_nd_1);  constant_pad_nd = constant_pad_nd_1 = None
        permute: "bf16[128100, 4096]" = torch.ops.aten.permute.default(view_4, [1, 0])
        constant_pad_nd_2: "bf16[128104, 4096]" = torch.ops.aten.constant_pad_nd.default(permute, _shape_param_7);  permute = _shape_param_7 = None
        mm_1: "bf16[128104, 1536]" = torch.ops.aten.mm.default(constant_pad_nd_2, arg8_1);  constant_pad_nd_2 = arg8_1 = None
        slice_1: "bf16[128100, 1536]" = torch.ops.aten.slice.Tensor(mm_1, 0, 0, -4);  mm_1 = None
        sum_2: "f32[1, 128100]" = torch.ops.aten.sum.dim_IntList(view_4, [0], True, dtype = torch.float32);  view_4 = None
        view_5: "f32[128100]" = torch.ops.aten.view.default(sum_2, _shape_param_8);  sum_2 = _shape_param_8 = None
        convert_element_type_6: "bf16[128100]" = torch.ops.prims.convert_element_type.default(view_5, torch.bfloat16);  view_5 = None
        view_6: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm, _shape_param_9);  mm = _shape_param_9 = None
        convert_element_type_7: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_6, torch.float32);  view_6 = None
        convert_element_type_8: "f32[128100, 1536]" = torch.ops.prims.convert_element_type.default(slice_1, torch.float32);  slice_1 = None
        convert_element_type_9: "f32[128100]" = torch.ops.prims.convert_element_type.default(convert_element_type_6, torch.float32);  convert_element_type_6 = None
        mul_2: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_7, arg9_1);  arg9_1 = None
        mul_3: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_2, 1536)
        sum_3: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_2, [2], True)
        view_7: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(arg10_1, _shape_param_10);  arg10_1 = _shape_param_10 = None
        convert_element_type_10: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_7, torch.float32);  view_7 = None
        mul_4: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_10, 0.5)
        mul_5: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_10, 0.7071067811865476)
        erf: "f32[8, 512, 1536]" = torch.ops.aten.erf.default(mul_5);  mul_5 = None
        add_1: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_6: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_4, add_1);  mul_4 = None
        convert_element_type_11: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(mul_6, torch.bfloat16);  mul_6 = None
        convert_element_type_12: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_11, torch.float32);  convert_element_type_11 = None
        sub_3: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(convert_element_type_12, arg11_1);  convert_element_type_12 = arg11_1 = None
        mul_7: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_3, arg12_1);  sub_3 = None
        mul_8: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_2, mul_7);  mul_2 = None
        sum_4: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_8, [2], True);  mul_8 = None
        mul_9: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_7, sum_4);  sum_4 = None
        sub_4: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_3, sum_3);  mul_3 = sum_3 = None
        sub_5: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_4, mul_9);  sub_4 = mul_9 = None
        div_1: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(arg12_1, 1536);  arg12_1 = None
        mul_10: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(div_1, sub_5);  div_1 = sub_5 = None
        mul_11: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_7, mul_7);  mul_7 = None
        sum_5: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_11, [0, 1]);  mul_11 = None
        sum_6: "f32[1536]" = torch.ops.aten.sum.dim_IntList(convert_element_type_7, [0, 1]);  convert_element_type_7 = None
        convert_element_type_13: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(mul_10, torch.bfloat16);  mul_10 = None
        convert_element_type_14: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_13, torch.float32);  convert_element_type_13 = None
        mul_12: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_1, 0.5);  add_1 = None
        mul_13: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_10, convert_element_type_10)
        mul_14: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_13, -0.5);  mul_13 = None
        exp_1: "f32[8, 512, 1536]" = torch.ops.aten.exp.default(mul_14);  mul_14 = None
        mul_15: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(exp_1, 0.3989422804014327);  exp_1 = None
        mul_16: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_10, mul_15);  convert_element_type_10 = mul_15 = None
        add_2: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_12, mul_16);  mul_12 = mul_16 = None
        mul_17: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_14, add_2);  convert_element_type_14 = add_2 = None
        convert_element_type_15: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(mul_17, torch.bfloat16);  mul_17 = None
        view_8: "bf16[4096, 1536]" = torch.ops.aten.view.default(convert_element_type_15, _shape_param_11);  convert_element_type_15 = _shape_param_11 = None
        mm_2: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_8, arg13_1);  arg13_1 = None
        permute_1: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_8, [1, 0])
        mm_3: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_1, arg14_1);  permute_1 = arg14_1 = None
        sum_7: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_8, [0], True, dtype = torch.float32);  view_8 = None
        view_9: "f32[1536]" = torch.ops.aten.view.default(sum_7, _shape_param_12);  sum_7 = _shape_param_12 = None
        convert_element_type_16: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_9, torch.bfloat16);  view_9 = None
        view_10: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_2, _shape_param_13);  mm_2 = _shape_param_13 = None
        convert_element_type_17: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_10, torch.float32);  view_10 = None
        convert_element_type_18: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        convert_element_type_19: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_16, torch.float32);  convert_element_type_16 = None
        mul_18: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_17, arg15_1);  arg15_1 = None
        mul_19: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_18, 1536)
        sum_8: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_18, [2], True)
        mul_20: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_18, arg16_1);  mul_18 = None
        sum_9: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_20, [2], True);  mul_20 = None
        mul_21: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg16_1, sum_9);  sum_9 = None
        sub_6: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_19, sum_8);  mul_19 = sum_8 = None
        sub_7: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_6, mul_21);  sub_6 = mul_21 = None
        mul_22: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg17_1, sub_7);  arg17_1 = sub_7 = None
        mul_23: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_17, arg16_1);  arg16_1 = None
        sum_10: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_23, [0, 1]);  mul_23 = None
        sum_11: "f32[1536]" = torch.ops.aten.sum.dim_IntList(convert_element_type_17, [0, 1]);  convert_element_type_17 = None
        convert_element_type_20: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(mul_22, torch.bfloat16)
        convert_element_type_21: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(arg18_1, torch.bfloat16);  arg18_1 = None
        mul_24: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_21, 1.1111111111111112);  convert_element_type_21 = None
        mul_25: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_20, mul_24);  convert_element_type_20 = mul_24 = None
        view_11: "bf16[4096, 1536]" = torch.ops.aten.view.default(mul_25, _shape_param_14);  mul_25 = _shape_param_14 = None
        mm_4: "bf16[4096, 6144]" = torch.ops.aten.mm.default(view_11, arg19_1);  arg19_1 = None
        permute_2: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_11, [1, 0])
        mm_5: "bf16[1536, 6144]" = torch.ops.aten.mm.default(permute_2, arg20_1);  permute_2 = arg20_1 = None
        sum_12: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_11, [0], True, dtype = torch.float32);  view_11 = None
        view_12: "f32[1536]" = torch.ops.aten.view.default(sum_12, _shape_param_15);  sum_12 = _shape_param_15 = None
        convert_element_type_22: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_12, torch.bfloat16);  view_12 = None
        view_13: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(mm_4, _shape_param_16);  mm_4 = _shape_param_16 = None
        convert_element_type_23: "f32[1536, 6144]" = torch.ops.prims.convert_element_type.default(mm_5, torch.float32);  mm_5 = None
        convert_element_type_24: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_22, torch.float32);  convert_element_type_22 = None
        convert_element_type_25: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_13, torch.float32);  view_13 = None
        view_14: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(arg21_1, _shape_param_17);  arg21_1 = _shape_param_17 = None
        convert_element_type_26: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_14, torch.float32);  view_14 = None
        mul_26: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_26, 0.7071067811865476)
        erf_1: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_26);  mul_26 = None
        add_3: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_27: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(add_3, 0.5);  add_3 = None
        mul_28: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_26, convert_element_type_26)
        mul_29: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_28, -0.5);  mul_28 = None
        exp_2: "f32[8, 512, 6144]" = torch.ops.aten.exp.default(mul_29);  mul_29 = None
        mul_30: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_2, 0.3989422804014327);  exp_2 = None
        mul_31: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_26, mul_30);  convert_element_type_26 = mul_30 = None
        add_4: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(mul_27, mul_31);  mul_27 = mul_31 = None
        mul_32: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_25, add_4);  convert_element_type_25 = add_4 = None
        convert_element_type_27: "bf16[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(mul_32, torch.bfloat16);  mul_32 = None
        view_15: "bf16[4096, 6144]" = torch.ops.aten.view.default(convert_element_type_27, _shape_param_18);  convert_element_type_27 = _shape_param_18 = None
        mm_6: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_15, arg22_1);  arg22_1 = None
        permute_3: "bf16[6144, 4096]" = torch.ops.aten.permute.default(view_15, [1, 0])
        mm_7: "bf16[6144, 1536]" = torch.ops.aten.mm.default(permute_3, arg23_1);  permute_3 = arg23_1 = None
        sum_13: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_15, [0], True, dtype = torch.float32);  view_15 = None
        view_16: "f32[6144]" = torch.ops.aten.view.default(sum_13, _shape_param_19);  sum_13 = _shape_param_19 = None
        convert_element_type_28: "bf16[6144]" = torch.ops.prims.convert_element_type.default(view_16, torch.bfloat16);  view_16 = None
        view_17: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_6, _shape_param_20);  mm_6 = _shape_param_20 = None
        convert_element_type_29: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_17, torch.float32);  view_17 = None
        add_5: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_22, convert_element_type_29);  mul_22 = convert_element_type_29 = None
        convert_element_type_30: "f32[6144, 1536]" = torch.ops.prims.convert_element_type.default(mm_7, torch.float32);  mm_7 = None
        convert_element_type_31: "f32[6144]" = torch.ops.prims.convert_element_type.default(convert_element_type_28, torch.float32);  convert_element_type_28 = None
        mul_33: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_5, arg24_1);  arg24_1 = None
        mul_34: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_33, 1536)
        sum_14: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_33, [2], True)
        mul_35: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_33, arg25_1);  mul_33 = None
        sum_15: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_35, [2], True);  mul_35 = None
        mul_36: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg25_1, sum_15);  sum_15 = None
        sub_8: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_34, sum_14);  mul_34 = sum_14 = None
        sub_9: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_8, mul_36);  sub_8 = mul_36 = None
        mul_37: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg26_1, sub_9);  arg26_1 = sub_9 = None
        mul_38: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_5, arg25_1);  arg25_1 = None
        sum_16: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_38, [0, 1]);  mul_38 = None
        sum_17: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_5, [0, 1]);  add_5 = None
        convert_element_type_32: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(mul_37, torch.bfloat16)
        convert_element_type_33: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(arg27_1, torch.bfloat16);  arg27_1 = None
        mul_39: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_33, 1.1111111111111112);  convert_element_type_33 = None
        mul_40: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_32, mul_39);  convert_element_type_32 = mul_39 = None
        view_18: "bf16[4096, 1536]" = torch.ops.aten.view.default(mul_40, _shape_param_21);  mul_40 = _shape_param_21 = None
        mm_8: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_18, arg28_1);  arg28_1 = None
        permute_4: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_18, [1, 0])
        mm_9: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_4, arg29_1);  permute_4 = arg29_1 = None
        sum_18: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_18, [0], True, dtype = torch.float32);  view_18 = None
        view_19: "f32[1536]" = torch.ops.aten.view.default(sum_18, _shape_param_22);  sum_18 = _shape_param_22 = None
        convert_element_type_34: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_19, torch.bfloat16);  view_19 = None
        view_20: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_8, _shape_param_23);  mm_8 = _shape_param_23 = None
        convert_element_type_35: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_9, torch.float32);  mm_9 = None
        convert_element_type_36: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_34, torch.float32);  convert_element_type_34 = None
        view_21: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_20, _shape_param_24);  view_20 = _shape_param_24 = None
        permute_5: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_21, [0, 2, 1, 3]);  view_21 = None
        clone: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
        view_22: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone, _shape_param_25);  clone = _shape_param_25 = None
        bmm: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(arg30_1, view_22);  arg30_1 = None
        bmm_1: "bf16[192, 512, 512]" = torch.ops.aten.bmm.default(view_22, arg31_1);  view_22 = arg31_1 = None
        convert_element_type_37: "f32[192, 512, 512]" = torch.ops.prims.convert_element_type.default(bmm_1, torch.float32);  bmm_1 = None
        view_23: "f32[8, 24, 512, 512]" = torch.ops.aten.view.default(convert_element_type_37, _shape_param_26);  convert_element_type_37 = _shape_param_26 = None
        convert_element_type_38: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(arg32_1, torch.float32);  arg32_1 = None
        mul_41: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_38, 1.1111111111111112);  convert_element_type_38 = None
        mul_42: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(view_23, mul_41);  view_23 = mul_41 = None
        convert_element_type_39: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(arg33_1, torch.float32);  arg33_1 = None
        sub_10: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_39, arg34_1);  convert_element_type_39 = arg34_1 = None
        exp_3: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        div_2: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_3, arg35_1);  exp_3 = arg35_1 = None
        mul_43: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_42, div_2);  mul_42 = None
        sum_19: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_43, [-1], True)
        neg: "f32[8, 24, 512, 512]" = torch.ops.aten.neg.default(div_2);  div_2 = None
        fma: "f32[8, 24, 512, 512]" = torch.ops.prims.fma.default(neg, sum_19, mul_43);  neg = sum_19 = mul_43 = None
        convert_element_type_40: "bf16[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None
        full_2: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_3: "b8[8, 1, 512, 512]" = torch.ops.aten.full.default(_shape_param_27, False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_27 = None
        where_3: "bf16[8, 24, 512, 512]" = torch.ops.aten.where.self(full_3, full_2, convert_element_type_40);  convert_element_type_40 = None
        view_24: "bf16[192, 512, 512]" = torch.ops.aten.view.default(where_3, _shape_param_28);  where_3 = _shape_param_28 = None
        bmm_2: "bf16[192, 64, 512]" = torch.ops.aten.bmm.default(arg36_1, view_24);  arg36_1 = None
        bmm_3: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(view_24, arg37_1);  view_24 = arg37_1 = None
        full_4: "bf16[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        div_3: "bf16[192, 64, 512]" = torch.ops.aten.div.Tensor(bmm_2, full_4);  bmm_2 = None
        permute_6: "bf16[192, 512, 64]" = torch.ops.aten.permute.default(div_3, [0, 2, 1]);  div_3 = None
        view_25: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm, _shape_param_29);  bmm = _shape_param_29 = None
        permute_7: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_25, [0, 2, 1, 3]);  view_25 = None
        clone_1: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
        view_26: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_1, _shape_param_30);  clone_1 = _shape_param_30 = None
        view_27: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_26, _shape_param_31);  view_26 = _shape_param_31 = None
        mm_10: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_27, arg38_1);  arg38_1 = None
        permute_8: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_27, [1, 0])
        mm_11: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_8, arg39_1);  permute_8 = None
        sum_20: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_27, [0], True, dtype = torch.float32);  view_27 = None
        view_28: "f32[1536]" = torch.ops.aten.view.default(sum_20, _shape_param_32);  sum_20 = _shape_param_32 = None
        convert_element_type_41: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_28, torch.bfloat16);  view_28 = None
        view_29: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_10, _shape_param_33);  mm_10 = _shape_param_33 = None
        convert_element_type_42: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_29, torch.float32);  view_29 = None
        add_6: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_37, convert_element_type_42);  mul_37 = convert_element_type_42 = None
        convert_element_type_43: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_11, torch.float32);  mm_11 = None
        convert_element_type_44: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_41, torch.float32);  convert_element_type_41 = None
        view_30: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(permute_6, _shape_param_34);  permute_6 = _shape_param_34 = None
        permute_9: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None
        view_31: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(permute_9, _shape_param_35);  permute_9 = _shape_param_35 = None
        clone_2: "bf16[8, 512, 1536]" = torch.ops.aten.clone.default(view_31, memory_format = torch.contiguous_format);  view_31 = None
        view_32: "bf16[4096, 1536]" = torch.ops.aten.view.default(clone_2, _shape_param_36);  clone_2 = _shape_param_36 = None
        mm_12: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_32, arg40_1);  arg40_1 = None
        permute_10: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_32, [1, 0])
        mm_13: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_10, arg39_1);  permute_10 = None
        sum_21: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_32, [0], True, dtype = torch.float32);  view_32 = None
        view_33: "f32[1536]" = torch.ops.aten.view.default(sum_21, _shape_param_37);  sum_21 = _shape_param_37 = None
        convert_element_type_45: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_33, torch.bfloat16);  view_33 = None
        view_34: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_12, _shape_param_38);  mm_12 = _shape_param_38 = None
        convert_element_type_46: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_34, torch.float32);  view_34 = None
        add_7: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_6, convert_element_type_46);  add_6 = convert_element_type_46 = None
        convert_element_type_47: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_13, torch.float32);  mm_13 = None
        convert_element_type_48: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_45, torch.float32);  convert_element_type_45 = None
        view_35: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_3, _shape_param_39);  bmm_3 = _shape_param_39 = None
        permute_11: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_35, [0, 2, 1, 3]);  view_35 = None
        clone_3: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_11, memory_format = torch.contiguous_format);  permute_11 = None
        view_36: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_3, _shape_param_40);  clone_3 = _shape_param_40 = None
        view_37: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_36, _shape_param_41);  view_36 = _shape_param_41 = None
        mm_14: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_37, arg41_1);  arg41_1 = None
        permute_12: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_37, [1, 0])
        mm_15: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_12, arg39_1);  permute_12 = arg39_1 = None
        sum_22: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_37, [0], True, dtype = torch.float32);  view_37 = None
        view_38: "f32[1536]" = torch.ops.aten.view.default(sum_22, _shape_param_42);  sum_22 = _shape_param_42 = None
        convert_element_type_49: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_38, torch.bfloat16);  view_38 = None
        view_39: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_14, _shape_param_43);  mm_14 = _shape_param_43 = None
        convert_element_type_50: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_39, torch.float32);  view_39 = None
        add_8: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_7, convert_element_type_50);  add_7 = convert_element_type_50 = None
        convert_element_type_51: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_15, torch.float32);  mm_15 = None
        convert_element_type_52: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_49, torch.float32);  convert_element_type_49 = None
        mul_44: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_8, arg42_1);  arg42_1 = None
        mul_45: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_44, 1536)
        sum_23: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_44, [2], True)
        mul_46: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_44, arg43_1);  mul_44 = None
        sum_24: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_46, [2], True);  mul_46 = None
        mul_47: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg43_1, sum_24);  sum_24 = None
        sub_11: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_45, sum_23);  mul_45 = sum_23 = None
        sub_12: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_11, mul_47);  sub_11 = mul_47 = None
        mul_48: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg44_1, sub_12);  arg44_1 = sub_12 = None
        mul_49: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_8, arg43_1);  arg43_1 = None
        sum_25: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_49, [0, 1]);  mul_49 = None
        sum_26: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_8, [0, 1]);  add_8 = None
        convert_element_type_53: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(mul_48, torch.bfloat16)
        convert_element_type_54: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(arg45_1, torch.bfloat16);  arg45_1 = None
        mul_50: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_54, 1.1111111111111112);  convert_element_type_54 = None
        mul_51: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_53, mul_50);  convert_element_type_53 = mul_50 = None
        view_40: "bf16[4096, 1536]" = torch.ops.aten.view.default(mul_51, _shape_param_44);  mul_51 = _shape_param_44 = None
        mm_16: "bf16[4096, 6144]" = torch.ops.aten.mm.default(view_40, arg46_1);  arg46_1 = None
        permute_13: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_40, [1, 0])
        mm_17: "bf16[1536, 6144]" = torch.ops.aten.mm.default(permute_13, arg47_1);  permute_13 = arg47_1 = None
        sum_27: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_40, [0], True, dtype = torch.float32);  view_40 = None
        view_41: "f32[1536]" = torch.ops.aten.view.default(sum_27, _shape_param_45);  sum_27 = _shape_param_45 = None
        convert_element_type_55: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_41, torch.bfloat16);  view_41 = None
        view_42: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(mm_16, _shape_param_46);  mm_16 = _shape_param_46 = None
        convert_element_type_56: "f32[1536, 6144]" = torch.ops.prims.convert_element_type.default(mm_17, torch.float32);  mm_17 = None
        convert_element_type_57: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_55, torch.float32);  convert_element_type_55 = None
        convert_element_type_58: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_42, torch.float32);  view_42 = None
        view_43: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(arg48_1, _shape_param_47);  arg48_1 = _shape_param_47 = None
        convert_element_type_59: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_43, torch.float32);  view_43 = None
        mul_52: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_59, 0.7071067811865476)
        erf_2: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_52);  mul_52 = None
        add_9: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_53: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(add_9, 0.5);  add_9 = None
        mul_54: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_59, convert_element_type_59)
        mul_55: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_54, -0.5);  mul_54 = None
        exp_4: "f32[8, 512, 6144]" = torch.ops.aten.exp.default(mul_55);  mul_55 = None
        mul_56: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_4, 0.3989422804014327);  exp_4 = None
        mul_57: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_59, mul_56);  convert_element_type_59 = mul_56 = None
        add_10: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(mul_53, mul_57);  mul_53 = mul_57 = None
        mul_58: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_58, add_10);  convert_element_type_58 = add_10 = None
        convert_element_type_60: "bf16[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(mul_58, torch.bfloat16);  mul_58 = None
        view_44: "bf16[4096, 6144]" = torch.ops.aten.view.default(convert_element_type_60, _shape_param_48);  convert_element_type_60 = _shape_param_48 = None
        mm_18: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_44, arg49_1);  arg49_1 = None
        permute_14: "bf16[6144, 4096]" = torch.ops.aten.permute.default(view_44, [1, 0])
        mm_19: "bf16[6144, 1536]" = torch.ops.aten.mm.default(permute_14, arg50_1);  permute_14 = arg50_1 = None
        sum_28: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_44, [0], True, dtype = torch.float32);  view_44 = None
        view_45: "f32[6144]" = torch.ops.aten.view.default(sum_28, _shape_param_49);  sum_28 = _shape_param_49 = None
        convert_element_type_61: "bf16[6144]" = torch.ops.prims.convert_element_type.default(view_45, torch.bfloat16);  view_45 = None
        view_46: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_18, _shape_param_50);  mm_18 = _shape_param_50 = None
        convert_element_type_62: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_46, torch.float32);  view_46 = None
        add_11: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_48, convert_element_type_62);  mul_48 = convert_element_type_62 = None
        convert_element_type_63: "f32[6144, 1536]" = torch.ops.prims.convert_element_type.default(mm_19, torch.float32);  mm_19 = None
        convert_element_type_64: "f32[6144]" = torch.ops.prims.convert_element_type.default(convert_element_type_61, torch.float32);  convert_element_type_61 = None
        mul_59: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_11, arg51_1);  arg51_1 = None
        mul_60: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_59, 1536)
        sum_29: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_59, [2], True)
        mul_61: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_59, arg52_1);  mul_59 = None
        sum_30: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_61, [2], True);  mul_61 = None
        mul_62: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg52_1, sum_30);  sum_30 = None
        sub_13: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_60, sum_29);  mul_60 = sum_29 = None
        sub_14: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_13, mul_62);  sub_13 = mul_62 = None
        mul_63: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg53_1, sub_14);  arg53_1 = sub_14 = None
        mul_64: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_11, arg52_1);  arg52_1 = None
        sum_31: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_64, [0, 1]);  mul_64 = None
        sum_32: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_11, [0, 1]);  add_11 = None
        convert_element_type_65: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(mul_63, torch.bfloat16)
        convert_element_type_66: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(arg54_1, torch.bfloat16);  arg54_1 = None
        mul_65: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_66, 1.1111111111111112);  convert_element_type_66 = None
        mul_66: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_65, mul_65);  convert_element_type_65 = mul_65 = None
        view_47: "bf16[4096, 1536]" = torch.ops.aten.view.default(mul_66, _shape_param_51);  mul_66 = _shape_param_51 = None
        mm_20: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_47, arg55_1);  arg55_1 = None
        permute_15: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_47, [1, 0])
        mm_21: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_15, arg56_1);  permute_15 = arg56_1 = None
        sum_33: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_47, [0], True, dtype = torch.float32);  view_47 = None
        view_48: "f32[1536]" = torch.ops.aten.view.default(sum_33, _shape_param_52);  sum_33 = _shape_param_52 = None
        convert_element_type_67: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_48, torch.bfloat16);  view_48 = None
        view_49: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_20, _shape_param_53);  mm_20 = _shape_param_53 = None
        convert_element_type_68: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_21, torch.float32);  mm_21 = None
        convert_element_type_69: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_67, torch.float32);  convert_element_type_67 = None
        view_50: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_49, _shape_param_54);  view_49 = _shape_param_54 = None
        permute_16: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_50, [0, 2, 1, 3]);  view_50 = None
        clone_4: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
        view_51: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_4, _shape_param_55);  clone_4 = _shape_param_55 = None
        bmm_4: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(arg57_1, view_51);  arg57_1 = None
        bmm_5: "bf16[192, 512, 512]" = torch.ops.aten.bmm.default(view_51, arg58_1);  view_51 = arg58_1 = None
        convert_element_type_70: "f32[192, 512, 512]" = torch.ops.prims.convert_element_type.default(bmm_5, torch.float32);  bmm_5 = None
        view_52: "f32[8, 24, 512, 512]" = torch.ops.aten.view.default(convert_element_type_70, _shape_param_56);  convert_element_type_70 = _shape_param_56 = None
        convert_element_type_71: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(arg59_1, torch.float32);  arg59_1 = None
        mul_67: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_71, 1.1111111111111112);  convert_element_type_71 = None
        mul_68: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(view_52, mul_67);  view_52 = mul_67 = None
        convert_element_type_72: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(arg60_1, torch.float32);  arg60_1 = None
        sub_15: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_72, arg61_1);  convert_element_type_72 = arg61_1 = None
        exp_5: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_15);  sub_15 = None
        div_4: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_5, arg62_1);  exp_5 = arg62_1 = None
        mul_69: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_68, div_4);  mul_68 = None
        sum_34: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_69, [-1], True)
        neg_1: "f32[8, 24, 512, 512]" = torch.ops.aten.neg.default(div_4);  div_4 = None
        fma_1: "f32[8, 24, 512, 512]" = torch.ops.prims.fma.default(neg_1, sum_34, mul_69);  neg_1 = sum_34 = mul_69 = None
        convert_element_type_73: "bf16[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(fma_1, torch.bfloat16);  fma_1 = None
        where_4: "bf16[8, 24, 512, 512]" = torch.ops.aten.where.self(full_3, full_2, convert_element_type_73);  convert_element_type_73 = None
        view_53: "bf16[192, 512, 512]" = torch.ops.aten.view.default(where_4, _shape_param_57);  where_4 = _shape_param_57 = None
        bmm_6: "bf16[192, 64, 512]" = torch.ops.aten.bmm.default(arg63_1, view_53);  arg63_1 = None
        bmm_7: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(view_53, arg64_1);  view_53 = arg64_1 = None
        div_5: "bf16[192, 64, 512]" = torch.ops.aten.div.Tensor(bmm_6, full_4);  bmm_6 = None
        permute_17: "bf16[192, 512, 64]" = torch.ops.aten.permute.default(div_5, [0, 2, 1]);  div_5 = None
        view_54: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_4, _shape_param_58);  bmm_4 = _shape_param_58 = None
        permute_18: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None
        clone_5: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
        view_55: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_5, _shape_param_59);  clone_5 = _shape_param_59 = None
        view_56: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_55, _shape_param_60);  view_55 = _shape_param_60 = None
        mm_22: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_56, arg65_1);  arg65_1 = None
        permute_19: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_56, [1, 0])
        mm_23: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_19, arg66_1);  permute_19 = None
        sum_35: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_56, [0], True, dtype = torch.float32);  view_56 = None
        view_57: "f32[1536]" = torch.ops.aten.view.default(sum_35, _shape_param_61);  sum_35 = _shape_param_61 = None
        convert_element_type_74: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_57, torch.bfloat16);  view_57 = None
        view_58: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_22, _shape_param_62);  mm_22 = _shape_param_62 = None
        convert_element_type_75: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_58, torch.float32);  view_58 = None
        add_12: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_63, convert_element_type_75);  mul_63 = convert_element_type_75 = None
        convert_element_type_76: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_23, torch.float32);  mm_23 = None
        convert_element_type_77: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_74, torch.float32);  convert_element_type_74 = None
        view_59: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(permute_17, _shape_param_63);  permute_17 = _shape_param_63 = None
        permute_20: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_59, [0, 2, 1, 3]);  view_59 = None
        view_60: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(permute_20, _shape_param_64);  permute_20 = _shape_param_64 = None
        clone_6: "bf16[8, 512, 1536]" = torch.ops.aten.clone.default(view_60, memory_format = torch.contiguous_format);  view_60 = None
        view_61: "bf16[4096, 1536]" = torch.ops.aten.view.default(clone_6, _shape_param_65);  clone_6 = _shape_param_65 = None
        mm_24: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_61, arg67_1);  arg67_1 = None
        permute_21: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_61, [1, 0])
        mm_25: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_21, arg66_1);  permute_21 = None
        sum_36: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_61, [0], True, dtype = torch.float32);  view_61 = None
        view_62: "f32[1536]" = torch.ops.aten.view.default(sum_36, _shape_param_66);  sum_36 = _shape_param_66 = None
        convert_element_type_78: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_62, torch.bfloat16);  view_62 = None
        view_63: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_24, _shape_param_67);  mm_24 = _shape_param_67 = None
        convert_element_type_79: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_63, torch.float32);  view_63 = None
        add_13: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_12, convert_element_type_79);  add_12 = convert_element_type_79 = None
        convert_element_type_80: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_25, torch.float32);  mm_25 = None
        convert_element_type_81: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_78, torch.float32);  convert_element_type_78 = None
        view_64: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_7, _shape_param_68);  bmm_7 = _shape_param_68 = None
        permute_22: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_64, [0, 2, 1, 3]);  view_64 = None
        clone_7: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_22, memory_format = torch.contiguous_format);  permute_22 = None
        view_65: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_7, _shape_param_69);  clone_7 = _shape_param_69 = None
        view_66: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_65, _shape_param_70);  view_65 = _shape_param_70 = None
        mm_26: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_66, arg68_1);  arg68_1 = None
        permute_23: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_66, [1, 0])
        mm_27: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_23, arg66_1);  permute_23 = arg66_1 = None
        sum_37: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_66, [0], True, dtype = torch.float32);  view_66 = None
        view_67: "f32[1536]" = torch.ops.aten.view.default(sum_37, _shape_param_71);  sum_37 = _shape_param_71 = None
        convert_element_type_82: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_67, torch.bfloat16);  view_67 = None
        view_68: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_26, _shape_param_72);  mm_26 = _shape_param_72 = None
        convert_element_type_83: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_68, torch.float32);  view_68 = None
        add_14: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_13, convert_element_type_83);  add_13 = convert_element_type_83 = None
        convert_element_type_84: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_27, torch.float32);  mm_27 = None
        convert_element_type_85: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_82, torch.float32);  convert_element_type_82 = None
        mul_70: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_14, arg69_1);  arg69_1 = None
        mul_71: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_70, 1536)
        sum_38: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_70, [2], True)
        mul_72: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_70, arg70_1);  mul_70 = None
        sum_39: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_72, [2], True);  mul_72 = None
        mul_73: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg70_1, sum_39);  sum_39 = None
        sub_16: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_71, sum_38);  mul_71 = sum_38 = None
        sub_17: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_16, mul_73);  sub_16 = mul_73 = None
        mul_74: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg71_1, sub_17);  arg71_1 = sub_17 = None
        mul_75: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_14, arg70_1);  arg70_1 = None
        sum_40: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_75, [0, 1]);  mul_75 = None
        sum_41: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_14, [0, 1]);  add_14 = None
        convert_element_type_86: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(mul_74, torch.bfloat16)
        convert_element_type_87: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(arg72_1, torch.bfloat16);  arg72_1 = None
        mul_76: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_87, 1.1111111111111112);  convert_element_type_87 = None
        mul_77: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_86, mul_76);  convert_element_type_86 = mul_76 = None
        view_69: "bf16[4096, 1536]" = torch.ops.aten.view.default(mul_77, _shape_param_73);  mul_77 = _shape_param_73 = None
        mm_28: "bf16[4096, 6144]" = torch.ops.aten.mm.default(view_69, arg73_1);  arg73_1 = None
        permute_24: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_69, [1, 0])
        mm_29: "bf16[1536, 6144]" = torch.ops.aten.mm.default(permute_24, arg74_1);  permute_24 = arg74_1 = None
        sum_42: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_69, [0], True, dtype = torch.float32);  view_69 = None
        view_70: "f32[1536]" = torch.ops.aten.view.default(sum_42, _shape_param_74);  sum_42 = _shape_param_74 = None
        convert_element_type_88: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_70, torch.bfloat16);  view_70 = None
        view_71: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(mm_28, _shape_param_75);  mm_28 = _shape_param_75 = None
        convert_element_type_89: "f32[1536, 6144]" = torch.ops.prims.convert_element_type.default(mm_29, torch.float32);  mm_29 = None
        convert_element_type_90: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_88, torch.float32);  convert_element_type_88 = None
        convert_element_type_91: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_71, torch.float32);  view_71 = None
        view_72: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(arg75_1, _shape_param_76);  arg75_1 = _shape_param_76 = None
        convert_element_type_92: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_72, torch.float32);  view_72 = None
        mul_78: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_92, 0.7071067811865476)
        erf_3: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_78);  mul_78 = None
        add_15: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_79: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(add_15, 0.5);  add_15 = None
        mul_80: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_92, convert_element_type_92)
        mul_81: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_80, -0.5);  mul_80 = None
        exp_6: "f32[8, 512, 6144]" = torch.ops.aten.exp.default(mul_81);  mul_81 = None
        mul_82: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_6, 0.3989422804014327);  exp_6 = None
        mul_83: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_92, mul_82);  convert_element_type_92 = mul_82 = None
        add_16: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(mul_79, mul_83);  mul_79 = mul_83 = None
        mul_84: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_91, add_16);  convert_element_type_91 = add_16 = None
        convert_element_type_93: "bf16[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(mul_84, torch.bfloat16);  mul_84 = None
        view_73: "bf16[4096, 6144]" = torch.ops.aten.view.default(convert_element_type_93, _shape_param_77);  convert_element_type_93 = _shape_param_77 = None
        mm_30: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_73, arg76_1);  arg76_1 = None
        permute_25: "bf16[6144, 4096]" = torch.ops.aten.permute.default(view_73, [1, 0])
        mm_31: "bf16[6144, 1536]" = torch.ops.aten.mm.default(permute_25, arg77_1);  permute_25 = arg77_1 = None
        sum_43: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_73, [0], True, dtype = torch.float32);  view_73 = None
        view_74: "f32[6144]" = torch.ops.aten.view.default(sum_43, _shape_param_78);  sum_43 = _shape_param_78 = None
        convert_element_type_94: "bf16[6144]" = torch.ops.prims.convert_element_type.default(view_74, torch.bfloat16);  view_74 = None
        view_75: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_30, _shape_param_79);  mm_30 = _shape_param_79 = None
        convert_element_type_95: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_75, torch.float32);  view_75 = None
        add_17: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_74, convert_element_type_95);  mul_74 = convert_element_type_95 = None
        convert_element_type_96: "f32[6144, 1536]" = torch.ops.prims.convert_element_type.default(mm_31, torch.float32);  mm_31 = None
        convert_element_type_97: "f32[6144]" = torch.ops.prims.convert_element_type.default(convert_element_type_94, torch.float32);  convert_element_type_94 = None
        mul_85: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_17, arg78_1);  arg78_1 = None
        mul_86: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_85, 1536)
        sum_44: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_85, [2], True)
        mul_87: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_85, arg79_1);  mul_85 = None
        sum_45: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_87, [2], True);  mul_87 = None
        mul_88: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg79_1, sum_45);  sum_45 = None
        sub_18: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_86, sum_44);  mul_86 = sum_44 = None
        sub_19: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_18, mul_88);  sub_18 = mul_88 = None
        mul_89: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg80_1, sub_19);  arg80_1 = sub_19 = None
        mul_90: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_17, arg79_1);  arg79_1 = None
        sum_46: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_90, [0, 1]);  mul_90 = None
        sum_47: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_17, [0, 1]);  add_17 = None
        convert_element_type_98: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(mul_89, torch.bfloat16)
        convert_element_type_99: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(arg81_1, torch.bfloat16);  arg81_1 = None
        mul_91: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_99, 1.1111111111111112);  convert_element_type_99 = None
        mul_92: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_98, mul_91);  convert_element_type_98 = mul_91 = None
        view_76: "bf16[4096, 1536]" = torch.ops.aten.view.default(mul_92, _shape_param_80);  mul_92 = _shape_param_80 = None
        mm_32: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_76, arg82_1);  arg82_1 = None
        permute_26: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_76, [1, 0])
        mm_33: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_26, arg83_1);  permute_26 = arg83_1 = None
        sum_48: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_76, [0], True, dtype = torch.float32);  view_76 = None
        view_77: "f32[1536]" = torch.ops.aten.view.default(sum_48, _shape_param_81);  sum_48 = _shape_param_81 = None
        convert_element_type_100: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_77, torch.bfloat16);  view_77 = None
        view_78: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_32, _shape_param_82);  mm_32 = _shape_param_82 = None
        convert_element_type_101: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_33, torch.float32);  mm_33 = None
        convert_element_type_102: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_100, torch.float32);  convert_element_type_100 = None
        view_79: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_78, _shape_param_83);  view_78 = _shape_param_83 = None
        permute_27: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_79, [0, 2, 1, 3]);  view_79 = None
        clone_8: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None
        view_80: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_8, _shape_param_84);  clone_8 = _shape_param_84 = None
        bmm_8: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(arg84_1, view_80);  arg84_1 = None
        bmm_9: "bf16[192, 512, 512]" = torch.ops.aten.bmm.default(view_80, arg85_1);  view_80 = arg85_1 = None
        convert_element_type_103: "f32[192, 512, 512]" = torch.ops.prims.convert_element_type.default(bmm_9, torch.float32);  bmm_9 = None
        view_81: "f32[8, 24, 512, 512]" = torch.ops.aten.view.default(convert_element_type_103, _shape_param_85);  convert_element_type_103 = _shape_param_85 = None
        convert_element_type_104: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(arg86_1, torch.float32);  arg86_1 = None
        mul_93: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_104, 1.1111111111111112);  convert_element_type_104 = None
        mul_94: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(view_81, mul_93);  view_81 = mul_93 = None
        convert_element_type_105: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(arg87_1, torch.float32);  arg87_1 = None
        sub_20: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_105, arg88_1);  convert_element_type_105 = arg88_1 = None
        exp_7: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_20);  sub_20 = None
        div_6: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_7, arg89_1);  exp_7 = arg89_1 = None
        mul_95: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_94, div_6);  mul_94 = None
        sum_49: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_95, [-1], True)
        neg_2: "f32[8, 24, 512, 512]" = torch.ops.aten.neg.default(div_6);  div_6 = None
        fma_2: "f32[8, 24, 512, 512]" = torch.ops.prims.fma.default(neg_2, sum_49, mul_95);  neg_2 = sum_49 = mul_95 = None
        convert_element_type_106: "bf16[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(fma_2, torch.bfloat16);  fma_2 = None
        where_5: "bf16[8, 24, 512, 512]" = torch.ops.aten.where.self(full_3, full_2, convert_element_type_106);  convert_element_type_106 = None
        view_82: "bf16[192, 512, 512]" = torch.ops.aten.view.default(where_5, _shape_param_86);  where_5 = _shape_param_86 = None
        bmm_10: "bf16[192, 64, 512]" = torch.ops.aten.bmm.default(arg90_1, view_82);  arg90_1 = None
        bmm_11: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(view_82, arg91_1);  view_82 = arg91_1 = None
        div_7: "bf16[192, 64, 512]" = torch.ops.aten.div.Tensor(bmm_10, full_4);  bmm_10 = None
        permute_28: "bf16[192, 512, 64]" = torch.ops.aten.permute.default(div_7, [0, 2, 1]);  div_7 = None
        view_83: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_8, _shape_param_87);  bmm_8 = _shape_param_87 = None
        permute_29: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_83, [0, 2, 1, 3]);  view_83 = None
        clone_9: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
        view_84: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_9, _shape_param_88);  clone_9 = _shape_param_88 = None
        view_85: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_84, _shape_param_89);  view_84 = _shape_param_89 = None
        mm_34: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_85, arg92_1);  arg92_1 = None
        permute_30: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_85, [1, 0])
        mm_35: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_30, arg93_1);  permute_30 = None
        sum_50: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_85, [0], True, dtype = torch.float32);  view_85 = None
        view_86: "f32[1536]" = torch.ops.aten.view.default(sum_50, _shape_param_90);  sum_50 = _shape_param_90 = None
        convert_element_type_107: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_86, torch.bfloat16);  view_86 = None
        view_87: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_34, _shape_param_91);  mm_34 = _shape_param_91 = None
        convert_element_type_108: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_87, torch.float32);  view_87 = None
        add_18: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_89, convert_element_type_108);  mul_89 = convert_element_type_108 = None
        convert_element_type_109: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_35, torch.float32);  mm_35 = None
        convert_element_type_110: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_107, torch.float32);  convert_element_type_107 = None
        view_88: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(permute_28, _shape_param_92);  permute_28 = _shape_param_92 = None
        permute_31: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_88, [0, 2, 1, 3]);  view_88 = None
        view_89: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(permute_31, _shape_param_93);  permute_31 = _shape_param_93 = None
        clone_10: "bf16[8, 512, 1536]" = torch.ops.aten.clone.default(view_89, memory_format = torch.contiguous_format);  view_89 = None
        view_90: "bf16[4096, 1536]" = torch.ops.aten.view.default(clone_10, _shape_param_94);  clone_10 = _shape_param_94 = None
        mm_36: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_90, arg94_1);  arg94_1 = None
        permute_32: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_90, [1, 0])
        mm_37: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_32, arg93_1);  permute_32 = None
        sum_51: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_90, [0], True, dtype = torch.float32);  view_90 = None
        view_91: "f32[1536]" = torch.ops.aten.view.default(sum_51, _shape_param_95);  sum_51 = _shape_param_95 = None
        convert_element_type_111: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_91, torch.bfloat16);  view_91 = None
        view_92: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_36, _shape_param_96);  mm_36 = _shape_param_96 = None
        convert_element_type_112: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_92, torch.float32);  view_92 = None
        add_19: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_18, convert_element_type_112);  add_18 = convert_element_type_112 = None
        convert_element_type_113: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_37, torch.float32);  mm_37 = None
        convert_element_type_114: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_111, torch.float32);  convert_element_type_111 = None
        view_93: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_11, _shape_param_97);  bmm_11 = _shape_param_97 = None
        permute_33: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_93, [0, 2, 1, 3]);  view_93 = None
        clone_11: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_33, memory_format = torch.contiguous_format);  permute_33 = None
        view_94: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_11, _shape_param_98);  clone_11 = _shape_param_98 = None
        view_95: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_94, _shape_param_99);  view_94 = _shape_param_99 = None
        mm_38: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_95, arg95_1);  arg95_1 = None
        permute_34: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_95, [1, 0])
        mm_39: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_34, arg93_1);  permute_34 = arg93_1 = None
        sum_52: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_95, [0], True, dtype = torch.float32);  view_95 = None
        view_96: "f32[1536]" = torch.ops.aten.view.default(sum_52, _shape_param_100);  sum_52 = _shape_param_100 = None
        convert_element_type_115: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_96, torch.bfloat16);  view_96 = None
        view_97: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_38, _shape_param_101);  mm_38 = _shape_param_101 = None
        convert_element_type_116: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_97, torch.float32);  view_97 = None
        add_20: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_19, convert_element_type_116);  add_19 = convert_element_type_116 = None
        convert_element_type_117: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_39, torch.float32);  mm_39 = None
        convert_element_type_118: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_115, torch.float32);  convert_element_type_115 = None
        mul_96: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_20, arg96_1);  arg96_1 = None
        mul_97: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_96, 1536)
        sum_53: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_96, [2], True)
        mul_98: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_96, arg97_1);  mul_96 = None
        sum_54: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_98, [2], True);  mul_98 = None
        mul_99: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg97_1, sum_54);  sum_54 = None
        sub_21: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_97, sum_53);  mul_97 = sum_53 = None
        sub_22: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_21, mul_99);  sub_21 = mul_99 = None
        mul_100: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg98_1, sub_22);  arg98_1 = sub_22 = None
        mul_101: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_20, arg97_1);  arg97_1 = None
        sum_55: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_101, [0, 1]);  mul_101 = None
        sum_56: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_20, [0, 1]);  add_20 = None
        convert_element_type_119: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(mul_100, torch.bfloat16)
        convert_element_type_120: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(arg99_1, torch.bfloat16);  arg99_1 = None
        mul_102: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_120, 1.1111111111111112);  convert_element_type_120 = None
        mul_103: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_119, mul_102);  convert_element_type_119 = mul_102 = None
        view_98: "bf16[4096, 1536]" = torch.ops.aten.view.default(mul_103, _shape_param_102);  mul_103 = _shape_param_102 = None
        mm_40: "bf16[4096, 6144]" = torch.ops.aten.mm.default(view_98, arg100_1);  arg100_1 = None
        permute_35: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_98, [1, 0])
        mm_41: "bf16[1536, 6144]" = torch.ops.aten.mm.default(permute_35, arg101_1);  permute_35 = arg101_1 = None
        sum_57: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_98, [0], True, dtype = torch.float32);  view_98 = None
        view_99: "f32[1536]" = torch.ops.aten.view.default(sum_57, _shape_param_103);  sum_57 = _shape_param_103 = None
        convert_element_type_121: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_99, torch.bfloat16);  view_99 = None
        view_100: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(mm_40, _shape_param_104);  mm_40 = _shape_param_104 = None
        convert_element_type_122: "f32[1536, 6144]" = torch.ops.prims.convert_element_type.default(mm_41, torch.float32);  mm_41 = None
        convert_element_type_123: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_121, torch.float32);  convert_element_type_121 = None
        convert_element_type_124: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_100, torch.float32);  view_100 = None
        view_101: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(arg102_1, _shape_param_105);  arg102_1 = _shape_param_105 = None
        convert_element_type_125: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_101, torch.float32);  view_101 = None
        mul_104: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_125, 0.7071067811865476)
        erf_4: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_104);  mul_104 = None
        add_21: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_105: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(add_21, 0.5);  add_21 = None
        mul_106: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_125, convert_element_type_125)
        mul_107: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_106, -0.5);  mul_106 = None
        exp_8: "f32[8, 512, 6144]" = torch.ops.aten.exp.default(mul_107);  mul_107 = None
        mul_108: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_8, 0.3989422804014327);  exp_8 = None
        mul_109: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_125, mul_108);  convert_element_type_125 = mul_108 = None
        add_22: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(mul_105, mul_109);  mul_105 = mul_109 = None
        mul_110: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_124, add_22);  convert_element_type_124 = add_22 = None
        convert_element_type_126: "bf16[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(mul_110, torch.bfloat16);  mul_110 = None
        view_102: "bf16[4096, 6144]" = torch.ops.aten.view.default(convert_element_type_126, _shape_param_106);  convert_element_type_126 = _shape_param_106 = None
        mm_42: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_102, arg103_1);  arg103_1 = None
        permute_36: "bf16[6144, 4096]" = torch.ops.aten.permute.default(view_102, [1, 0])
        mm_43: "bf16[6144, 1536]" = torch.ops.aten.mm.default(permute_36, arg104_1);  permute_36 = arg104_1 = None
        sum_58: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_102, [0], True, dtype = torch.float32);  view_102 = None
        view_103: "f32[6144]" = torch.ops.aten.view.default(sum_58, _shape_param_107);  sum_58 = _shape_param_107 = None
        convert_element_type_127: "bf16[6144]" = torch.ops.prims.convert_element_type.default(view_103, torch.bfloat16);  view_103 = None
        view_104: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_42, _shape_param_108);  mm_42 = _shape_param_108 = None
        convert_element_type_128: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_104, torch.float32);  view_104 = None
        add_23: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_100, convert_element_type_128);  mul_100 = convert_element_type_128 = None
        convert_element_type_129: "f32[6144, 1536]" = torch.ops.prims.convert_element_type.default(mm_43, torch.float32);  mm_43 = None
        convert_element_type_130: "f32[6144]" = torch.ops.prims.convert_element_type.default(convert_element_type_127, torch.float32);  convert_element_type_127 = None
        mul_111: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_23, arg105_1);  arg105_1 = None
        mul_112: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_111, 1536)
        sum_59: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_111, [2], True)
        mul_113: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_111, arg106_1);  mul_111 = None
        sum_60: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_113, [2], True);  mul_113 = None
        mul_114: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg106_1, sum_60);  sum_60 = None
        sub_23: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_112, sum_59);  mul_112 = sum_59 = None
        sub_24: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_23, mul_114);  sub_23 = mul_114 = None
        mul_115: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg107_1, sub_24);  arg107_1 = sub_24 = None
        mul_116: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_23, arg106_1);  arg106_1 = None
        sum_61: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_116, [0, 1]);  mul_116 = None
        sum_62: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_23, [0, 1]);  add_23 = None
        convert_element_type_131: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(mul_115, torch.bfloat16)
        convert_element_type_132: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(arg108_1, torch.bfloat16);  arg108_1 = None
        mul_117: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_132, 1.1111111111111112);  convert_element_type_132 = None
        mul_118: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_131, mul_117);  convert_element_type_131 = mul_117 = None
        view_105: "bf16[4096, 1536]" = torch.ops.aten.view.default(mul_118, _shape_param_109);  mul_118 = _shape_param_109 = None
        mm_44: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_105, arg109_1);  arg109_1 = None
        permute_37: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_105, [1, 0])
        mm_45: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_37, arg110_1);  permute_37 = arg110_1 = None
        sum_63: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_105, [0], True, dtype = torch.float32);  view_105 = None
        view_106: "f32[1536]" = torch.ops.aten.view.default(sum_63, _shape_param_110);  sum_63 = _shape_param_110 = None
        convert_element_type_133: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_106, torch.bfloat16);  view_106 = None
        view_107: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_44, _shape_param_111);  mm_44 = _shape_param_111 = None
        convert_element_type_134: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_45, torch.float32);  mm_45 = None
        convert_element_type_135: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_133, torch.float32);  convert_element_type_133 = None
        view_108: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_107, _shape_param_112);  view_107 = _shape_param_112 = None
        permute_38: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_108, [0, 2, 1, 3]);  view_108 = None
        clone_12: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_38, memory_format = torch.contiguous_format);  permute_38 = None
        view_109: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_12, _shape_param_113);  clone_12 = _shape_param_113 = None
        bmm_12: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(arg111_1, view_109);  arg111_1 = None
        bmm_13: "bf16[192, 512, 512]" = torch.ops.aten.bmm.default(view_109, arg112_1);  view_109 = arg112_1 = None
        convert_element_type_136: "f32[192, 512, 512]" = torch.ops.prims.convert_element_type.default(bmm_13, torch.float32);  bmm_13 = None
        view_110: "f32[8, 24, 512, 512]" = torch.ops.aten.view.default(convert_element_type_136, _shape_param_114);  convert_element_type_136 = _shape_param_114 = None
        convert_element_type_137: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(arg113_1, torch.float32);  arg113_1 = None
        mul_119: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_137, 1.1111111111111112);  convert_element_type_137 = None
        mul_120: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(view_110, mul_119);  view_110 = mul_119 = None
        convert_element_type_138: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(arg114_1, torch.float32);  arg114_1 = None
        sub_25: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_138, arg115_1);  convert_element_type_138 = arg115_1 = None
        exp_9: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
        div_8: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_9, arg116_1);  exp_9 = arg116_1 = None
        mul_121: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_120, div_8);  mul_120 = None
        sum_64: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_121, [-1], True)
        neg_3: "f32[8, 24, 512, 512]" = torch.ops.aten.neg.default(div_8);  div_8 = None
        fma_3: "f32[8, 24, 512, 512]" = torch.ops.prims.fma.default(neg_3, sum_64, mul_121);  neg_3 = sum_64 = mul_121 = None
        convert_element_type_139: "bf16[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(fma_3, torch.bfloat16);  fma_3 = None
        where_6: "bf16[8, 24, 512, 512]" = torch.ops.aten.where.self(full_3, full_2, convert_element_type_139);  convert_element_type_139 = None
        view_111: "bf16[192, 512, 512]" = torch.ops.aten.view.default(where_6, _shape_param_115);  where_6 = _shape_param_115 = None
        bmm_14: "bf16[192, 64, 512]" = torch.ops.aten.bmm.default(arg117_1, view_111);  arg117_1 = None
        bmm_15: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(view_111, arg118_1);  view_111 = arg118_1 = None
        div_9: "bf16[192, 64, 512]" = torch.ops.aten.div.Tensor(bmm_14, full_4);  bmm_14 = None
        permute_39: "bf16[192, 512, 64]" = torch.ops.aten.permute.default(div_9, [0, 2, 1]);  div_9 = None
        view_112: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_12, _shape_param_116);  bmm_12 = _shape_param_116 = None
        permute_40: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_112, [0, 2, 1, 3]);  view_112 = None
        clone_13: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
        view_113: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_13, _shape_param_117);  clone_13 = _shape_param_117 = None
        view_114: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_113, _shape_param_118);  view_113 = _shape_param_118 = None
        mm_46: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_114, arg119_1);  arg119_1 = None
        permute_41: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_114, [1, 0])
        mm_47: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_41, arg120_1);  permute_41 = None
        sum_65: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_114, [0], True, dtype = torch.float32);  view_114 = None
        view_115: "f32[1536]" = torch.ops.aten.view.default(sum_65, _shape_param_119);  sum_65 = _shape_param_119 = None
        convert_element_type_140: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_115, torch.bfloat16);  view_115 = None
        view_116: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_46, _shape_param_120);  mm_46 = _shape_param_120 = None
        convert_element_type_141: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_116, torch.float32);  view_116 = None
        add_24: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_115, convert_element_type_141);  mul_115 = convert_element_type_141 = None
        convert_element_type_142: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_47, torch.float32);  mm_47 = None
        convert_element_type_143: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_140, torch.float32);  convert_element_type_140 = None
        view_117: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(permute_39, _shape_param_121);  permute_39 = _shape_param_121 = None
        permute_42: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_117, [0, 2, 1, 3]);  view_117 = None
        view_118: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(permute_42, _shape_param_122);  permute_42 = _shape_param_122 = None
        clone_14: "bf16[8, 512, 1536]" = torch.ops.aten.clone.default(view_118, memory_format = torch.contiguous_format);  view_118 = None
        view_119: "bf16[4096, 1536]" = torch.ops.aten.view.default(clone_14, _shape_param_123);  clone_14 = _shape_param_123 = None
        mm_48: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_119, arg121_1);  arg121_1 = None
        permute_43: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_119, [1, 0])
        mm_49: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_43, arg120_1);  permute_43 = None
        sum_66: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_119, [0], True, dtype = torch.float32);  view_119 = None
        view_120: "f32[1536]" = torch.ops.aten.view.default(sum_66, _shape_param_124);  sum_66 = _shape_param_124 = None
        convert_element_type_144: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_120, torch.bfloat16);  view_120 = None
        view_121: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_48, _shape_param_125);  mm_48 = _shape_param_125 = None
        convert_element_type_145: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_121, torch.float32);  view_121 = None
        add_25: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_24, convert_element_type_145);  add_24 = convert_element_type_145 = None
        convert_element_type_146: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_49, torch.float32);  mm_49 = None
        convert_element_type_147: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_144, torch.float32);  convert_element_type_144 = None
        view_122: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_15, _shape_param_126);  bmm_15 = _shape_param_126 = None
        permute_44: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_122, [0, 2, 1, 3]);  view_122 = None
        clone_15: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_44, memory_format = torch.contiguous_format);  permute_44 = None
        view_123: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_15, _shape_param_127);  clone_15 = _shape_param_127 = None
        view_124: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_123, _shape_param_128);  view_123 = _shape_param_128 = None
        mm_50: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_124, arg122_1);  arg122_1 = None
        permute_45: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_124, [1, 0])
        mm_51: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_45, arg120_1);  permute_45 = arg120_1 = None
        sum_67: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_124, [0], True, dtype = torch.float32);  view_124 = None
        view_125: "f32[1536]" = torch.ops.aten.view.default(sum_67, _shape_param_129);  sum_67 = _shape_param_129 = None
        convert_element_type_148: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_125, torch.bfloat16);  view_125 = None
        view_126: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_50, _shape_param_130);  mm_50 = _shape_param_130 = None
        convert_element_type_149: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_126, torch.float32);  view_126 = None
        add_26: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_25, convert_element_type_149);  add_25 = convert_element_type_149 = None
        convert_element_type_150: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_51, torch.float32);  mm_51 = None
        convert_element_type_151: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_148, torch.float32);  convert_element_type_148 = None
        mul_122: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_26, arg123_1);  arg123_1 = None
        mul_123: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_122, 1536)
        sum_68: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_122, [2], True)
        mul_124: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_122, arg124_1);  mul_122 = None
        sum_69: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_124, [2], True);  mul_124 = None
        mul_125: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg124_1, sum_69);  sum_69 = None
        sub_26: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_123, sum_68);  mul_123 = sum_68 = None
        sub_27: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_26, mul_125);  sub_26 = mul_125 = None
        mul_126: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg125_1, sub_27);  arg125_1 = sub_27 = None
        mul_127: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_26, arg124_1);  arg124_1 = None
        sum_70: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_127, [0, 1]);  mul_127 = None
        sum_71: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_26, [0, 1]);  add_26 = None
        convert_element_type_152: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(mul_126, torch.bfloat16)
        convert_element_type_153: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(arg126_1, torch.bfloat16);  arg126_1 = None
        mul_128: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_153, 1.1111111111111112);  convert_element_type_153 = None
        mul_129: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_152, mul_128);  convert_element_type_152 = mul_128 = None
        view_127: "bf16[4096, 1536]" = torch.ops.aten.view.default(mul_129, _shape_param_131);  mul_129 = _shape_param_131 = None
        mm_52: "bf16[4096, 6144]" = torch.ops.aten.mm.default(view_127, arg127_1);  arg127_1 = None
        permute_46: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_127, [1, 0])
        mm_53: "bf16[1536, 6144]" = torch.ops.aten.mm.default(permute_46, arg128_1);  permute_46 = arg128_1 = None
        sum_72: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_127, [0], True, dtype = torch.float32);  view_127 = None
        view_128: "f32[1536]" = torch.ops.aten.view.default(sum_72, _shape_param_132);  sum_72 = _shape_param_132 = None
        convert_element_type_154: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_128, torch.bfloat16);  view_128 = None
        view_129: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(mm_52, _shape_param_133);  mm_52 = _shape_param_133 = None
        convert_element_type_155: "f32[1536, 6144]" = torch.ops.prims.convert_element_type.default(mm_53, torch.float32);  mm_53 = None
        convert_element_type_156: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_154, torch.float32);  convert_element_type_154 = None
        convert_element_type_157: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_129, torch.float32);  view_129 = None
        view_130: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(arg129_1, _shape_param_134);  arg129_1 = _shape_param_134 = None
        convert_element_type_158: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_130, torch.float32);  view_130 = None
        mul_130: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_158, 0.7071067811865476)
        erf_5: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_130);  mul_130 = None
        add_27: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_131: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(add_27, 0.5);  add_27 = None
        mul_132: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_158, convert_element_type_158)
        mul_133: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_132, -0.5);  mul_132 = None
        exp_10: "f32[8, 512, 6144]" = torch.ops.aten.exp.default(mul_133);  mul_133 = None
        mul_134: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_10, 0.3989422804014327);  exp_10 = None
        mul_135: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_158, mul_134);  convert_element_type_158 = mul_134 = None
        add_28: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(mul_131, mul_135);  mul_131 = mul_135 = None
        mul_136: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_157, add_28);  convert_element_type_157 = add_28 = None
        convert_element_type_159: "bf16[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(mul_136, torch.bfloat16);  mul_136 = None
        view_131: "bf16[4096, 6144]" = torch.ops.aten.view.default(convert_element_type_159, _shape_param_135);  convert_element_type_159 = _shape_param_135 = None
        mm_54: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_131, arg130_1);  arg130_1 = None
        permute_47: "bf16[6144, 4096]" = torch.ops.aten.permute.default(view_131, [1, 0])
        mm_55: "bf16[6144, 1536]" = torch.ops.aten.mm.default(permute_47, arg131_1);  permute_47 = arg131_1 = None
        sum_73: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_131, [0], True, dtype = torch.float32);  view_131 = None
        view_132: "f32[6144]" = torch.ops.aten.view.default(sum_73, _shape_param_136);  sum_73 = _shape_param_136 = None
        convert_element_type_160: "bf16[6144]" = torch.ops.prims.convert_element_type.default(view_132, torch.bfloat16);  view_132 = None
        view_133: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_54, _shape_param_137);  mm_54 = _shape_param_137 = None
        convert_element_type_161: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_133, torch.float32);  view_133 = None
        add_29: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_126, convert_element_type_161);  mul_126 = convert_element_type_161 = None
        convert_element_type_162: "f32[6144, 1536]" = torch.ops.prims.convert_element_type.default(mm_55, torch.float32);  mm_55 = None
        convert_element_type_163: "f32[6144]" = torch.ops.prims.convert_element_type.default(convert_element_type_160, torch.float32);  convert_element_type_160 = None
        mul_137: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_29, arg132_1);  arg132_1 = None
        mul_138: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_137, 1536)
        sum_74: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_137, [2], True)
        mul_139: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_137, arg133_1);  mul_137 = None
        sum_75: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_139, [2], True);  mul_139 = None
        mul_140: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg133_1, sum_75);  sum_75 = None
        sub_28: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_138, sum_74);  mul_138 = sum_74 = None
        sub_29: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_28, mul_140);  sub_28 = mul_140 = None
        mul_141: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg134_1, sub_29);  arg134_1 = sub_29 = None
        mul_142: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_29, arg133_1);  arg133_1 = None
        sum_76: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_142, [0, 1]);  mul_142 = None
        sum_77: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_29, [0, 1]);  add_29 = None
        convert_element_type_164: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(mul_141, torch.bfloat16)
        convert_element_type_165: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(arg135_1, torch.bfloat16);  arg135_1 = None
        mul_143: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_165, 1.1111111111111112);  convert_element_type_165 = None
        mul_144: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_164, mul_143);  convert_element_type_164 = mul_143 = None
        view_134: "bf16[4096, 1536]" = torch.ops.aten.view.default(mul_144, _shape_param_138);  mul_144 = _shape_param_138 = None
        mm_56: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_134, arg136_1);  arg136_1 = None
        permute_48: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_134, [1, 0])
        mm_57: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_48, arg137_1);  permute_48 = arg137_1 = None
        sum_78: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_134, [0], True, dtype = torch.float32);  view_134 = None
        view_135: "f32[1536]" = torch.ops.aten.view.default(sum_78, _shape_param_139);  sum_78 = _shape_param_139 = None
        convert_element_type_166: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_135, torch.bfloat16);  view_135 = None
        view_136: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_56, _shape_param_140);  mm_56 = _shape_param_140 = None
        convert_element_type_167: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_57, torch.float32);  mm_57 = None
        convert_element_type_168: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_166, torch.float32);  convert_element_type_166 = None
        view_137: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_136, _shape_param_141);  view_136 = _shape_param_141 = None
        permute_49: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_137, [0, 2, 1, 3]);  view_137 = None
        clone_16: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
        view_138: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_16, _shape_param_142);  clone_16 = _shape_param_142 = None
        bmm_16: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(arg138_1, view_138);  arg138_1 = None
        bmm_17: "bf16[192, 512, 512]" = torch.ops.aten.bmm.default(view_138, arg139_1);  view_138 = arg139_1 = None
        convert_element_type_169: "f32[192, 512, 512]" = torch.ops.prims.convert_element_type.default(bmm_17, torch.float32);  bmm_17 = None
        view_139: "f32[8, 24, 512, 512]" = torch.ops.aten.view.default(convert_element_type_169, _shape_param_143);  convert_element_type_169 = _shape_param_143 = None
        convert_element_type_170: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(arg140_1, torch.float32);  arg140_1 = None
        mul_145: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_170, 1.1111111111111112);  convert_element_type_170 = None
        mul_146: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(view_139, mul_145);  view_139 = mul_145 = None
        convert_element_type_171: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(arg141_1, torch.float32);  arg141_1 = None
        sub_30: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_171, arg142_1);  convert_element_type_171 = arg142_1 = None
        exp_11: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_30);  sub_30 = None
        div_10: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_11, arg143_1);  exp_11 = arg143_1 = None
        mul_147: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_146, div_10);  mul_146 = None
        sum_79: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_147, [-1], True)
        neg_4: "f32[8, 24, 512, 512]" = torch.ops.aten.neg.default(div_10);  div_10 = None
        fma_4: "f32[8, 24, 512, 512]" = torch.ops.prims.fma.default(neg_4, sum_79, mul_147);  neg_4 = sum_79 = mul_147 = None
        convert_element_type_172: "bf16[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(fma_4, torch.bfloat16);  fma_4 = None
        where_7: "bf16[8, 24, 512, 512]" = torch.ops.aten.where.self(full_3, full_2, convert_element_type_172);  convert_element_type_172 = None
        view_140: "bf16[192, 512, 512]" = torch.ops.aten.view.default(where_7, _shape_param_144);  where_7 = _shape_param_144 = None
        bmm_18: "bf16[192, 64, 512]" = torch.ops.aten.bmm.default(arg144_1, view_140);  arg144_1 = None
        bmm_19: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(view_140, arg145_1);  view_140 = arg145_1 = None
        div_11: "bf16[192, 64, 512]" = torch.ops.aten.div.Tensor(bmm_18, full_4);  bmm_18 = None
        permute_50: "bf16[192, 512, 64]" = torch.ops.aten.permute.default(div_11, [0, 2, 1]);  div_11 = None
        view_141: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_16, _shape_param_145);  bmm_16 = _shape_param_145 = None
        permute_51: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_141, [0, 2, 1, 3]);  view_141 = None
        clone_17: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
        view_142: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_17, _shape_param_146);  clone_17 = _shape_param_146 = None
        view_143: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_142, _shape_param_147);  view_142 = _shape_param_147 = None
        mm_58: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_143, arg146_1);  arg146_1 = None
        permute_52: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_143, [1, 0])
        mm_59: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_52, arg147_1);  permute_52 = None
        sum_80: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_143, [0], True, dtype = torch.float32);  view_143 = None
        view_144: "f32[1536]" = torch.ops.aten.view.default(sum_80, _shape_param_148);  sum_80 = _shape_param_148 = None
        convert_element_type_173: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_144, torch.bfloat16);  view_144 = None
        view_145: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_58, _shape_param_149);  mm_58 = _shape_param_149 = None
        convert_element_type_174: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_145, torch.float32);  view_145 = None
        add_30: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_141, convert_element_type_174);  mul_141 = convert_element_type_174 = None
        convert_element_type_175: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_59, torch.float32);  mm_59 = None
        convert_element_type_176: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_173, torch.float32);  convert_element_type_173 = None
        view_146: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(permute_50, _shape_param_150);  permute_50 = _shape_param_150 = None
        permute_53: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_146, [0, 2, 1, 3]);  view_146 = None
        view_147: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(permute_53, _shape_param_151);  permute_53 = _shape_param_151 = None
        clone_18: "bf16[8, 512, 1536]" = torch.ops.aten.clone.default(view_147, memory_format = torch.contiguous_format);  view_147 = None
        view_148: "bf16[4096, 1536]" = torch.ops.aten.view.default(clone_18, _shape_param_152);  clone_18 = _shape_param_152 = None
        mm_60: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_148, arg148_1);  arg148_1 = None
        permute_54: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_148, [1, 0])
        mm_61: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_54, arg147_1);  permute_54 = None
        sum_81: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_148, [0], True, dtype = torch.float32);  view_148 = None
        view_149: "f32[1536]" = torch.ops.aten.view.default(sum_81, _shape_param_153);  sum_81 = _shape_param_153 = None
        convert_element_type_177: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_149, torch.bfloat16);  view_149 = None
        view_150: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_60, _shape_param_154);  mm_60 = _shape_param_154 = None
        convert_element_type_178: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_150, torch.float32);  view_150 = None
        add_31: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_30, convert_element_type_178);  add_30 = convert_element_type_178 = None
        convert_element_type_179: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_61, torch.float32);  mm_61 = None
        convert_element_type_180: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_177, torch.float32);  convert_element_type_177 = None
        view_151: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_19, _shape_param_155);  bmm_19 = _shape_param_155 = None
        permute_55: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_151, [0, 2, 1, 3]);  view_151 = None
        clone_19: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_55, memory_format = torch.contiguous_format);  permute_55 = None
        view_152: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_19, _shape_param_156);  clone_19 = _shape_param_156 = None
        view_153: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_152, _shape_param_157);  view_152 = _shape_param_157 = None
        mm_62: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_153, arg149_1);  arg149_1 = None
        permute_56: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_153, [1, 0])
        mm_63: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_56, arg147_1);  permute_56 = arg147_1 = None
        sum_82: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_153, [0], True, dtype = torch.float32);  view_153 = None
        view_154: "f32[1536]" = torch.ops.aten.view.default(sum_82, _shape_param_158);  sum_82 = _shape_param_158 = None
        convert_element_type_181: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_154, torch.bfloat16);  view_154 = None
        view_155: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_62, _shape_param_159);  mm_62 = _shape_param_159 = None
        convert_element_type_182: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_155, torch.float32);  view_155 = None
        add_32: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_31, convert_element_type_182);  add_31 = convert_element_type_182 = None
        convert_element_type_183: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_63, torch.float32);  mm_63 = None
        convert_element_type_184: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_181, torch.float32);  convert_element_type_181 = None
        mul_148: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_32, arg150_1);  arg150_1 = None
        mul_149: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_148, 1536)
        sum_83: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_148, [2], True)
        mul_150: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_148, arg151_1);  mul_148 = None
        sum_84: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_150, [2], True);  mul_150 = None
        mul_151: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg151_1, sum_84);  sum_84 = None
        sub_31: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_149, sum_83);  mul_149 = sum_83 = None
        sub_32: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_31, mul_151);  sub_31 = mul_151 = None
        mul_152: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg152_1, sub_32);  arg152_1 = sub_32 = None
        mul_153: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_32, arg151_1);  arg151_1 = None
        sum_85: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_153, [0, 1]);  mul_153 = None
        sum_86: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_32, [0, 1]);  add_32 = None
        convert_element_type_185: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(mul_152, torch.bfloat16)
        convert_element_type_186: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(arg153_1, torch.bfloat16);  arg153_1 = None
        mul_154: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_186, 1.1111111111111112);  convert_element_type_186 = None
        mul_155: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_185, mul_154);  convert_element_type_185 = mul_154 = None
        view_156: "bf16[4096, 1536]" = torch.ops.aten.view.default(mul_155, _shape_param_160);  mul_155 = _shape_param_160 = None
        mm_64: "bf16[4096, 6144]" = torch.ops.aten.mm.default(view_156, arg154_1);  arg154_1 = None
        permute_57: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_156, [1, 0])
        mm_65: "bf16[1536, 6144]" = torch.ops.aten.mm.default(permute_57, arg155_1);  permute_57 = arg155_1 = None
        sum_87: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_156, [0], True, dtype = torch.float32);  view_156 = None
        view_157: "f32[1536]" = torch.ops.aten.view.default(sum_87, _shape_param_161);  sum_87 = _shape_param_161 = None
        convert_element_type_187: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_157, torch.bfloat16);  view_157 = None
        view_158: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(mm_64, _shape_param_162);  mm_64 = _shape_param_162 = None
        convert_element_type_188: "f32[1536, 6144]" = torch.ops.prims.convert_element_type.default(mm_65, torch.float32);  mm_65 = None
        convert_element_type_189: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_187, torch.float32);  convert_element_type_187 = None
        convert_element_type_190: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_158, torch.float32);  view_158 = None
        view_159: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(arg156_1, _shape_param_163);  arg156_1 = _shape_param_163 = None
        convert_element_type_191: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_159, torch.float32);  view_159 = None
        mul_156: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_191, 0.7071067811865476)
        erf_6: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_156);  mul_156 = None
        add_33: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_157: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(add_33, 0.5);  add_33 = None
        mul_158: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_191, convert_element_type_191)
        mul_159: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_158, -0.5);  mul_158 = None
        exp_12: "f32[8, 512, 6144]" = torch.ops.aten.exp.default(mul_159);  mul_159 = None
        mul_160: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_12, 0.3989422804014327);  exp_12 = None
        mul_161: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_191, mul_160);  convert_element_type_191 = mul_160 = None
        add_34: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(mul_157, mul_161);  mul_157 = mul_161 = None
        mul_162: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_190, add_34);  convert_element_type_190 = add_34 = None
        convert_element_type_192: "bf16[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(mul_162, torch.bfloat16);  mul_162 = None
        view_160: "bf16[4096, 6144]" = torch.ops.aten.view.default(convert_element_type_192, _shape_param_164);  convert_element_type_192 = _shape_param_164 = None
        mm_66: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_160, arg157_1);  arg157_1 = None
        permute_58: "bf16[6144, 4096]" = torch.ops.aten.permute.default(view_160, [1, 0])
        mm_67: "bf16[6144, 1536]" = torch.ops.aten.mm.default(permute_58, arg158_1);  permute_58 = arg158_1 = None
        sum_88: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_160, [0], True, dtype = torch.float32);  view_160 = None
        view_161: "f32[6144]" = torch.ops.aten.view.default(sum_88, _shape_param_165);  sum_88 = _shape_param_165 = None
        convert_element_type_193: "bf16[6144]" = torch.ops.prims.convert_element_type.default(view_161, torch.bfloat16);  view_161 = None
        view_162: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_66, _shape_param_166);  mm_66 = _shape_param_166 = None
        convert_element_type_194: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_162, torch.float32);  view_162 = None
        add_35: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_152, convert_element_type_194);  mul_152 = convert_element_type_194 = None
        convert_element_type_195: "f32[6144, 1536]" = torch.ops.prims.convert_element_type.default(mm_67, torch.float32);  mm_67 = None
        convert_element_type_196: "f32[6144]" = torch.ops.prims.convert_element_type.default(convert_element_type_193, torch.float32);  convert_element_type_193 = None
        mul_163: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_35, arg159_1);  arg159_1 = None
        mul_164: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_163, 1536)
        sum_89: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_163, [2], True)
        mul_165: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_163, arg160_1);  mul_163 = None
        sum_90: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_165, [2], True);  mul_165 = None
        mul_166: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg160_1, sum_90);  sum_90 = None
        sub_33: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_164, sum_89);  mul_164 = sum_89 = None
        sub_34: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_33, mul_166);  sub_33 = mul_166 = None
        mul_167: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg161_1, sub_34);  arg161_1 = sub_34 = None
        mul_168: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_35, arg160_1);  arg160_1 = None
        sum_91: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_168, [0, 1]);  mul_168 = None
        sum_92: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_35, [0, 1]);  add_35 = None
        convert_element_type_197: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(mul_167, torch.bfloat16)
        convert_element_type_198: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(arg162_1, torch.bfloat16);  arg162_1 = None
        mul_169: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_198, 1.1111111111111112);  convert_element_type_198 = None
        mul_170: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_197, mul_169);  convert_element_type_197 = mul_169 = None
        view_163: "bf16[4096, 1536]" = torch.ops.aten.view.default(mul_170, _shape_param_167);  mul_170 = _shape_param_167 = None
        mm_68: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_163, arg163_1);  arg163_1 = None
        permute_59: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_163, [1, 0])
        mm_69: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_59, arg164_1);  permute_59 = arg164_1 = None
        sum_93: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_163, [0], True, dtype = torch.float32);  view_163 = None
        view_164: "f32[1536]" = torch.ops.aten.view.default(sum_93, _shape_param_168);  sum_93 = _shape_param_168 = None
        convert_element_type_199: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_164, torch.bfloat16);  view_164 = None
        view_165: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_68, _shape_param_169);  mm_68 = _shape_param_169 = None
        convert_element_type_200: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_69, torch.float32);  mm_69 = None
        convert_element_type_201: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_199, torch.float32);  convert_element_type_199 = None
        view_166: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_165, _shape_param_170);  view_165 = _shape_param_170 = None
        permute_60: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_166, [0, 2, 1, 3]);  view_166 = None
        clone_20: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_60, memory_format = torch.contiguous_format);  permute_60 = None
        view_167: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_20, _shape_param_171);  clone_20 = _shape_param_171 = None
        bmm_20: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(arg165_1, view_167);  arg165_1 = None
        bmm_21: "bf16[192, 512, 512]" = torch.ops.aten.bmm.default(view_167, arg166_1);  view_167 = arg166_1 = None
        convert_element_type_202: "f32[192, 512, 512]" = torch.ops.prims.convert_element_type.default(bmm_21, torch.float32);  bmm_21 = None
        view_168: "f32[8, 24, 512, 512]" = torch.ops.aten.view.default(convert_element_type_202, _shape_param_172);  convert_element_type_202 = _shape_param_172 = None
        convert_element_type_203: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(arg167_1, torch.float32);  arg167_1 = None
        mul_171: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_203, 1.1111111111111112);  convert_element_type_203 = None
        mul_172: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(view_168, mul_171);  view_168 = mul_171 = None
        convert_element_type_204: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(arg168_1, torch.float32);  arg168_1 = None
        sub_35: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_204, arg169_1);  convert_element_type_204 = arg169_1 = None
        exp_13: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_35);  sub_35 = None
        div_12: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_13, arg170_1);  exp_13 = arg170_1 = None
        mul_173: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_172, div_12);  mul_172 = None
        sum_94: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_173, [-1], True)
        neg_5: "f32[8, 24, 512, 512]" = torch.ops.aten.neg.default(div_12);  div_12 = None
        fma_5: "f32[8, 24, 512, 512]" = torch.ops.prims.fma.default(neg_5, sum_94, mul_173);  neg_5 = sum_94 = mul_173 = None
        convert_element_type_205: "bf16[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(fma_5, torch.bfloat16);  fma_5 = None
        where_8: "bf16[8, 24, 512, 512]" = torch.ops.aten.where.self(full_3, full_2, convert_element_type_205);  convert_element_type_205 = None
        view_169: "bf16[192, 512, 512]" = torch.ops.aten.view.default(where_8, _shape_param_173);  where_8 = _shape_param_173 = None
        bmm_22: "bf16[192, 64, 512]" = torch.ops.aten.bmm.default(arg171_1, view_169);  arg171_1 = None
        bmm_23: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(view_169, arg172_1);  view_169 = arg172_1 = None
        div_13: "bf16[192, 64, 512]" = torch.ops.aten.div.Tensor(bmm_22, full_4);  bmm_22 = None
        permute_61: "bf16[192, 512, 64]" = torch.ops.aten.permute.default(div_13, [0, 2, 1]);  div_13 = None
        view_170: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_20, _shape_param_174);  bmm_20 = _shape_param_174 = None
        permute_62: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_170, [0, 2, 1, 3]);  view_170 = None
        clone_21: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
        view_171: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_21, _shape_param_175);  clone_21 = _shape_param_175 = None
        view_172: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_171, _shape_param_176);  view_171 = _shape_param_176 = None
        mm_70: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_172, arg173_1);  arg173_1 = None
        permute_63: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_172, [1, 0])
        mm_71: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_63, arg174_1);  permute_63 = None
        sum_95: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_172, [0], True, dtype = torch.float32);  view_172 = None
        view_173: "f32[1536]" = torch.ops.aten.view.default(sum_95, _shape_param_177);  sum_95 = _shape_param_177 = None
        convert_element_type_206: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_173, torch.bfloat16);  view_173 = None
        view_174: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_70, _shape_param_178);  mm_70 = _shape_param_178 = None
        convert_element_type_207: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_174, torch.float32);  view_174 = None
        add_36: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_167, convert_element_type_207);  mul_167 = convert_element_type_207 = None
        convert_element_type_208: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_71, torch.float32);  mm_71 = None
        convert_element_type_209: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_206, torch.float32);  convert_element_type_206 = None
        view_175: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(permute_61, _shape_param_179);  permute_61 = _shape_param_179 = None
        permute_64: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_175, [0, 2, 1, 3]);  view_175 = None
        view_176: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(permute_64, _shape_param_180);  permute_64 = _shape_param_180 = None
        clone_22: "bf16[8, 512, 1536]" = torch.ops.aten.clone.default(view_176, memory_format = torch.contiguous_format);  view_176 = None
        view_177: "bf16[4096, 1536]" = torch.ops.aten.view.default(clone_22, _shape_param_181);  clone_22 = _shape_param_181 = None
        mm_72: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_177, arg175_1);  arg175_1 = None
        permute_65: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_177, [1, 0])
        mm_73: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_65, arg174_1);  permute_65 = None
        sum_96: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_177, [0], True, dtype = torch.float32);  view_177 = None
        view_178: "f32[1536]" = torch.ops.aten.view.default(sum_96, _shape_param_182);  sum_96 = _shape_param_182 = None
        convert_element_type_210: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_178, torch.bfloat16);  view_178 = None
        view_179: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_72, _shape_param_183);  mm_72 = _shape_param_183 = None
        convert_element_type_211: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_179, torch.float32);  view_179 = None
        add_37: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_36, convert_element_type_211);  add_36 = convert_element_type_211 = None
        convert_element_type_212: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_73, torch.float32);  mm_73 = None
        convert_element_type_213: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_210, torch.float32);  convert_element_type_210 = None
        view_180: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_23, _shape_param_184);  bmm_23 = _shape_param_184 = None
        permute_66: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_180, [0, 2, 1, 3]);  view_180 = None
        clone_23: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_66, memory_format = torch.contiguous_format);  permute_66 = None
        view_181: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_23, _shape_param_185);  clone_23 = _shape_param_185 = None
        view_182: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_181, _shape_param_186);  view_181 = _shape_param_186 = None
        mm_74: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_182, arg176_1);  arg176_1 = None
        permute_67: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_182, [1, 0])
        mm_75: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_67, arg174_1);  permute_67 = arg174_1 = None
        sum_97: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_182, [0], True, dtype = torch.float32);  view_182 = None
        view_183: "f32[1536]" = torch.ops.aten.view.default(sum_97, _shape_param_187);  sum_97 = _shape_param_187 = None
        convert_element_type_214: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_183, torch.bfloat16);  view_183 = None
        view_184: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_74, _shape_param_188);  mm_74 = _shape_param_188 = None
        convert_element_type_215: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_184, torch.float32);  view_184 = None
        add_38: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_37, convert_element_type_215);  add_37 = convert_element_type_215 = None
        convert_element_type_216: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_75, torch.float32);  mm_75 = None
        convert_element_type_217: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_214, torch.float32);  convert_element_type_214 = None
        mul_174: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_38, arg177_1);  arg177_1 = None
        mul_175: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_174, 1536)
        sum_98: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_174, [2], True)
        mul_176: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_174, arg178_1);  mul_174 = None
        sum_99: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_176, [2], True);  mul_176 = None
        mul_177: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg178_1, sum_99);  sum_99 = None
        sub_36: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_175, sum_98);  mul_175 = sum_98 = None
        sub_37: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_36, mul_177);  sub_36 = mul_177 = None
        mul_178: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg179_1, sub_37);  arg179_1 = sub_37 = None
        mul_179: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_38, arg178_1);  arg178_1 = None
        sum_100: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_179, [0, 1]);  mul_179 = None
        sum_101: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_38, [0, 1]);  add_38 = None
        convert_element_type_218: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(mul_178, torch.bfloat16)
        convert_element_type_219: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(arg180_1, torch.bfloat16);  arg180_1 = None
        mul_180: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_219, 1.1111111111111112);  convert_element_type_219 = None
        mul_181: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_218, mul_180);  convert_element_type_218 = mul_180 = None
        view_185: "bf16[4096, 1536]" = torch.ops.aten.view.default(mul_181, _shape_param_189);  mul_181 = _shape_param_189 = None
        mm_76: "bf16[4096, 6144]" = torch.ops.aten.mm.default(view_185, arg181_1);  arg181_1 = None
        permute_68: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_185, [1, 0])
        mm_77: "bf16[1536, 6144]" = torch.ops.aten.mm.default(permute_68, arg182_1);  permute_68 = arg182_1 = None
        sum_102: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_185, [0], True, dtype = torch.float32);  view_185 = None
        view_186: "f32[1536]" = torch.ops.aten.view.default(sum_102, _shape_param_190);  sum_102 = _shape_param_190 = None
        convert_element_type_220: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_186, torch.bfloat16);  view_186 = None
        view_187: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(mm_76, _shape_param_191);  mm_76 = _shape_param_191 = None
        convert_element_type_221: "f32[1536, 6144]" = torch.ops.prims.convert_element_type.default(mm_77, torch.float32);  mm_77 = None
        convert_element_type_222: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_220, torch.float32);  convert_element_type_220 = None
        convert_element_type_223: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_187, torch.float32);  view_187 = None
        view_188: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(arg183_1, _shape_param_192);  arg183_1 = _shape_param_192 = None
        convert_element_type_224: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_188, torch.float32);  view_188 = None
        mul_182: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_224, 0.7071067811865476)
        erf_7: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_182);  mul_182 = None
        add_39: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_183: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(add_39, 0.5);  add_39 = None
        mul_184: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_224, convert_element_type_224)
        mul_185: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_184, -0.5);  mul_184 = None
        exp_14: "f32[8, 512, 6144]" = torch.ops.aten.exp.default(mul_185);  mul_185 = None
        mul_186: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_14, 0.3989422804014327);  exp_14 = None
        mul_187: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_224, mul_186);  convert_element_type_224 = mul_186 = None
        add_40: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(mul_183, mul_187);  mul_183 = mul_187 = None
        mul_188: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_223, add_40);  convert_element_type_223 = add_40 = None
        convert_element_type_225: "bf16[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(mul_188, torch.bfloat16);  mul_188 = None
        view_189: "bf16[4096, 6144]" = torch.ops.aten.view.default(convert_element_type_225, _shape_param_193);  convert_element_type_225 = _shape_param_193 = None
        mm_78: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_189, arg184_1);  arg184_1 = None
        permute_69: "bf16[6144, 4096]" = torch.ops.aten.permute.default(view_189, [1, 0])
        mm_79: "bf16[6144, 1536]" = torch.ops.aten.mm.default(permute_69, arg185_1);  permute_69 = arg185_1 = None
        sum_103: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_189, [0], True, dtype = torch.float32);  view_189 = None
        view_190: "f32[6144]" = torch.ops.aten.view.default(sum_103, _shape_param_194);  sum_103 = _shape_param_194 = None
        convert_element_type_226: "bf16[6144]" = torch.ops.prims.convert_element_type.default(view_190, torch.bfloat16);  view_190 = None
        view_191: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_78, _shape_param_195);  mm_78 = _shape_param_195 = None
        convert_element_type_227: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_191, torch.float32);  view_191 = None
        add_41: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_178, convert_element_type_227);  mul_178 = convert_element_type_227 = None
        convert_element_type_228: "f32[6144, 1536]" = torch.ops.prims.convert_element_type.default(mm_79, torch.float32);  mm_79 = None
        convert_element_type_229: "f32[6144]" = torch.ops.prims.convert_element_type.default(convert_element_type_226, torch.float32);  convert_element_type_226 = None
        mul_189: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_41, arg186_1);  arg186_1 = None
        mul_190: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_189, 1536)
        sum_104: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_189, [2], True)
        mul_191: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_189, arg187_1);  mul_189 = None
        sum_105: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_191, [2], True);  mul_191 = None
        mul_192: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg187_1, sum_105);  sum_105 = None
        sub_38: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_190, sum_104);  mul_190 = sum_104 = None
        sub_39: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_38, mul_192);  sub_38 = mul_192 = None
        mul_193: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg188_1, sub_39);  arg188_1 = sub_39 = None
        mul_194: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_41, arg187_1);  arg187_1 = None
        sum_106: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_194, [0, 1]);  mul_194 = None
        sum_107: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_41, [0, 1]);  add_41 = None
        convert_element_type_230: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(mul_193, torch.bfloat16)
        convert_element_type_231: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(arg189_1, torch.bfloat16);  arg189_1 = None
        mul_195: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_231, 1.1111111111111112);  convert_element_type_231 = None
        mul_196: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_230, mul_195);  convert_element_type_230 = mul_195 = None
        view_192: "bf16[4096, 1536]" = torch.ops.aten.view.default(mul_196, _shape_param_196);  mul_196 = _shape_param_196 = None
        mm_80: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_192, arg190_1);  arg190_1 = None
        permute_70: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_192, [1, 0])
        mm_81: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_70, arg191_1);  permute_70 = arg191_1 = None
        sum_108: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_192, [0], True, dtype = torch.float32);  view_192 = None
        view_193: "f32[1536]" = torch.ops.aten.view.default(sum_108, _shape_param_197);  sum_108 = _shape_param_197 = None
        convert_element_type_232: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_193, torch.bfloat16);  view_193 = None
        view_194: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_80, _shape_param_198);  mm_80 = _shape_param_198 = None
        convert_element_type_233: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_81, torch.float32);  mm_81 = None
        convert_element_type_234: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_232, torch.float32);  convert_element_type_232 = None
        view_195: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_194, _shape_param_199);  view_194 = _shape_param_199 = None
        permute_71: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_195, [0, 2, 1, 3]);  view_195 = None
        clone_24: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_71, memory_format = torch.contiguous_format);  permute_71 = None
        view_196: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_24, _shape_param_200);  clone_24 = _shape_param_200 = None
        bmm_24: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(arg192_1, view_196);  arg192_1 = None
        bmm_25: "bf16[192, 512, 512]" = torch.ops.aten.bmm.default(view_196, arg193_1);  view_196 = arg193_1 = None
        convert_element_type_235: "f32[192, 512, 512]" = torch.ops.prims.convert_element_type.default(bmm_25, torch.float32);  bmm_25 = None
        view_197: "f32[8, 24, 512, 512]" = torch.ops.aten.view.default(convert_element_type_235, _shape_param_201);  convert_element_type_235 = _shape_param_201 = None
        convert_element_type_236: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(arg194_1, torch.float32);  arg194_1 = None
        mul_197: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_236, 1.1111111111111112);  convert_element_type_236 = None
        mul_198: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(view_197, mul_197);  view_197 = mul_197 = None
        convert_element_type_237: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(arg195_1, torch.float32);  arg195_1 = None
        sub_40: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_237, arg196_1);  convert_element_type_237 = arg196_1 = None
        exp_15: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_40);  sub_40 = None
        div_14: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_15, arg197_1);  exp_15 = arg197_1 = None
        mul_199: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_198, div_14);  mul_198 = None
        sum_109: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_199, [-1], True)
        neg_6: "f32[8, 24, 512, 512]" = torch.ops.aten.neg.default(div_14);  div_14 = None
        fma_6: "f32[8, 24, 512, 512]" = torch.ops.prims.fma.default(neg_6, sum_109, mul_199);  neg_6 = sum_109 = mul_199 = None
        convert_element_type_238: "bf16[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(fma_6, torch.bfloat16);  fma_6 = None
        where_9: "bf16[8, 24, 512, 512]" = torch.ops.aten.where.self(full_3, full_2, convert_element_type_238);  convert_element_type_238 = None
        view_198: "bf16[192, 512, 512]" = torch.ops.aten.view.default(where_9, _shape_param_202);  where_9 = _shape_param_202 = None
        bmm_26: "bf16[192, 64, 512]" = torch.ops.aten.bmm.default(arg198_1, view_198);  arg198_1 = None
        bmm_27: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(view_198, arg199_1);  view_198 = arg199_1 = None
        div_15: "bf16[192, 64, 512]" = torch.ops.aten.div.Tensor(bmm_26, full_4);  bmm_26 = None
        permute_72: "bf16[192, 512, 64]" = torch.ops.aten.permute.default(div_15, [0, 2, 1]);  div_15 = None
        view_199: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_24, _shape_param_203);  bmm_24 = _shape_param_203 = None
        permute_73: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_199, [0, 2, 1, 3]);  view_199 = None
        clone_25: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
        view_200: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_25, _shape_param_204);  clone_25 = _shape_param_204 = None
        view_201: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_200, _shape_param_205);  view_200 = _shape_param_205 = None
        mm_82: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_201, arg200_1);  arg200_1 = None
        permute_74: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_201, [1, 0])
        mm_83: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_74, arg201_1);  permute_74 = None
        sum_110: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_201, [0], True, dtype = torch.float32);  view_201 = None
        view_202: "f32[1536]" = torch.ops.aten.view.default(sum_110, _shape_param_206);  sum_110 = _shape_param_206 = None
        convert_element_type_239: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_202, torch.bfloat16);  view_202 = None
        view_203: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_82, _shape_param_207);  mm_82 = _shape_param_207 = None
        convert_element_type_240: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_203, torch.float32);  view_203 = None
        add_42: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_193, convert_element_type_240);  mul_193 = convert_element_type_240 = None
        convert_element_type_241: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_83, torch.float32);  mm_83 = None
        convert_element_type_242: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_239, torch.float32);  convert_element_type_239 = None
        view_204: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(permute_72, _shape_param_208);  permute_72 = _shape_param_208 = None
        permute_75: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_204, [0, 2, 1, 3]);  view_204 = None
        view_205: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(permute_75, _shape_param_209);  permute_75 = _shape_param_209 = None
        clone_26: "bf16[8, 512, 1536]" = torch.ops.aten.clone.default(view_205, memory_format = torch.contiguous_format);  view_205 = None
        view_206: "bf16[4096, 1536]" = torch.ops.aten.view.default(clone_26, _shape_param_210);  clone_26 = _shape_param_210 = None
        mm_84: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_206, arg202_1);  arg202_1 = None
        permute_76: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_206, [1, 0])
        mm_85: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_76, arg201_1);  permute_76 = None
        sum_111: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_206, [0], True, dtype = torch.float32);  view_206 = None
        view_207: "f32[1536]" = torch.ops.aten.view.default(sum_111, _shape_param_211);  sum_111 = _shape_param_211 = None
        convert_element_type_243: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_207, torch.bfloat16);  view_207 = None
        view_208: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_84, _shape_param_212);  mm_84 = _shape_param_212 = None
        convert_element_type_244: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_208, torch.float32);  view_208 = None
        add_43: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_42, convert_element_type_244);  add_42 = convert_element_type_244 = None
        convert_element_type_245: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_85, torch.float32);  mm_85 = None
        convert_element_type_246: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_243, torch.float32);  convert_element_type_243 = None
        view_209: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_27, _shape_param_213);  bmm_27 = _shape_param_213 = None
        permute_77: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_209, [0, 2, 1, 3]);  view_209 = None
        clone_27: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_77, memory_format = torch.contiguous_format);  permute_77 = None
        view_210: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_27, _shape_param_214);  clone_27 = _shape_param_214 = None
        view_211: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_210, _shape_param_215);  view_210 = _shape_param_215 = None
        mm_86: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_211, arg203_1);  arg203_1 = None
        permute_78: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_211, [1, 0])
        mm_87: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_78, arg201_1);  permute_78 = arg201_1 = None
        sum_112: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_211, [0], True, dtype = torch.float32);  view_211 = None
        view_212: "f32[1536]" = torch.ops.aten.view.default(sum_112, _shape_param_216);  sum_112 = _shape_param_216 = None
        convert_element_type_247: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_212, torch.bfloat16);  view_212 = None
        view_213: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_86, _shape_param_217);  mm_86 = _shape_param_217 = None
        convert_element_type_248: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_213, torch.float32);  view_213 = None
        add_44: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_43, convert_element_type_248);  add_43 = convert_element_type_248 = None
        convert_element_type_249: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_87, torch.float32);  mm_87 = None
        convert_element_type_250: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_247, torch.float32);  convert_element_type_247 = None
        mul_200: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_44, arg204_1);  arg204_1 = None
        mul_201: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_200, 1536)
        sum_113: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_200, [2], True)
        mul_202: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_200, arg205_1);  mul_200 = None
        sum_114: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_202, [2], True);  mul_202 = None
        mul_203: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg205_1, sum_114);  sum_114 = None
        sub_41: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_201, sum_113);  mul_201 = sum_113 = None
        sub_42: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_41, mul_203);  sub_41 = mul_203 = None
        mul_204: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg206_1, sub_42);  arg206_1 = sub_42 = None
        mul_205: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_44, arg205_1);  arg205_1 = None
        sum_115: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_205, [0, 1]);  mul_205 = None
        sum_116: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_44, [0, 1]);  add_44 = None
        convert_element_type_251: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(mul_204, torch.bfloat16)
        convert_element_type_252: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(arg207_1, torch.bfloat16);  arg207_1 = None
        mul_206: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_252, 1.1111111111111112);  convert_element_type_252 = None
        mul_207: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_251, mul_206);  convert_element_type_251 = mul_206 = None
        view_214: "bf16[4096, 1536]" = torch.ops.aten.view.default(mul_207, _shape_param_218);  mul_207 = _shape_param_218 = None
        mm_88: "bf16[4096, 6144]" = torch.ops.aten.mm.default(view_214, arg208_1);  arg208_1 = None
        permute_79: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_214, [1, 0])
        mm_89: "bf16[1536, 6144]" = torch.ops.aten.mm.default(permute_79, arg209_1);  permute_79 = arg209_1 = None
        sum_117: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_214, [0], True, dtype = torch.float32);  view_214 = None
        view_215: "f32[1536]" = torch.ops.aten.view.default(sum_117, _shape_param_219);  sum_117 = _shape_param_219 = None
        convert_element_type_253: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_215, torch.bfloat16);  view_215 = None
        view_216: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(mm_88, _shape_param_220);  mm_88 = _shape_param_220 = None
        convert_element_type_254: "f32[1536, 6144]" = torch.ops.prims.convert_element_type.default(mm_89, torch.float32);  mm_89 = None
        convert_element_type_255: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_253, torch.float32);  convert_element_type_253 = None
        convert_element_type_256: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_216, torch.float32);  view_216 = None
        view_217: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(arg210_1, _shape_param_221);  arg210_1 = _shape_param_221 = None
        convert_element_type_257: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_217, torch.float32);  view_217 = None
        mul_208: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_257, 0.7071067811865476)
        erf_8: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_208);  mul_208 = None
        add_45: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_209: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(add_45, 0.5);  add_45 = None
        mul_210: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_257, convert_element_type_257)
        mul_211: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_210, -0.5);  mul_210 = None
        exp_16: "f32[8, 512, 6144]" = torch.ops.aten.exp.default(mul_211);  mul_211 = None
        mul_212: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_16, 0.3989422804014327);  exp_16 = None
        mul_213: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_257, mul_212);  convert_element_type_257 = mul_212 = None
        add_46: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(mul_209, mul_213);  mul_209 = mul_213 = None
        mul_214: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_256, add_46);  convert_element_type_256 = add_46 = None
        convert_element_type_258: "bf16[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(mul_214, torch.bfloat16);  mul_214 = None
        view_218: "bf16[4096, 6144]" = torch.ops.aten.view.default(convert_element_type_258, _shape_param_222);  convert_element_type_258 = _shape_param_222 = None
        mm_90: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_218, arg211_1);  arg211_1 = None
        permute_80: "bf16[6144, 4096]" = torch.ops.aten.permute.default(view_218, [1, 0])
        mm_91: "bf16[6144, 1536]" = torch.ops.aten.mm.default(permute_80, arg212_1);  permute_80 = arg212_1 = None
        sum_118: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_218, [0], True, dtype = torch.float32);  view_218 = None
        view_219: "f32[6144]" = torch.ops.aten.view.default(sum_118, _shape_param_223);  sum_118 = _shape_param_223 = None
        convert_element_type_259: "bf16[6144]" = torch.ops.prims.convert_element_type.default(view_219, torch.bfloat16);  view_219 = None
        view_220: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_90, _shape_param_224);  mm_90 = _shape_param_224 = None
        convert_element_type_260: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_220, torch.float32);  view_220 = None
        add_47: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_204, convert_element_type_260);  mul_204 = convert_element_type_260 = None
        convert_element_type_261: "f32[6144, 1536]" = torch.ops.prims.convert_element_type.default(mm_91, torch.float32);  mm_91 = None
        convert_element_type_262: "f32[6144]" = torch.ops.prims.convert_element_type.default(convert_element_type_259, torch.float32);  convert_element_type_259 = None
        mul_215: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_47, arg213_1);  arg213_1 = None
        mul_216: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_215, 1536)
        sum_119: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_215, [2], True)
        mul_217: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_215, arg214_1);  mul_215 = None
        sum_120: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_217, [2], True);  mul_217 = None
        mul_218: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg214_1, sum_120);  sum_120 = None
        sub_43: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_216, sum_119);  mul_216 = sum_119 = None
        sub_44: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_43, mul_218);  sub_43 = mul_218 = None
        mul_219: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg215_1, sub_44);  arg215_1 = sub_44 = None
        mul_220: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_47, arg214_1);  arg214_1 = None
        sum_121: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_220, [0, 1]);  mul_220 = None
        sum_122: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_47, [0, 1]);  add_47 = None
        convert_element_type_263: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(mul_219, torch.bfloat16)
        convert_element_type_264: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(arg216_1, torch.bfloat16);  arg216_1 = None
        mul_221: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_264, 1.1111111111111112);  convert_element_type_264 = None
        mul_222: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_263, mul_221);  convert_element_type_263 = mul_221 = None
        view_221: "bf16[4096, 1536]" = torch.ops.aten.view.default(mul_222, _shape_param_225);  mul_222 = _shape_param_225 = None
        mm_92: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_221, arg217_1);  arg217_1 = None
        permute_81: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_221, [1, 0])
        mm_93: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_81, arg218_1);  permute_81 = arg218_1 = None
        sum_123: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_221, [0], True, dtype = torch.float32);  view_221 = None
        view_222: "f32[1536]" = torch.ops.aten.view.default(sum_123, _shape_param_226);  sum_123 = _shape_param_226 = None
        convert_element_type_265: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_222, torch.bfloat16);  view_222 = None
        view_223: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_92, _shape_param_227);  mm_92 = _shape_param_227 = None
        convert_element_type_266: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_93, torch.float32);  mm_93 = None
        convert_element_type_267: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_265, torch.float32);  convert_element_type_265 = None
        view_224: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_223, _shape_param_228);  view_223 = _shape_param_228 = None
        permute_82: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_224, [0, 2, 1, 3]);  view_224 = None
        clone_28: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
        view_225: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_28, _shape_param_229);  clone_28 = _shape_param_229 = None
        bmm_28: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(arg219_1, view_225);  arg219_1 = None
        bmm_29: "bf16[192, 512, 512]" = torch.ops.aten.bmm.default(view_225, arg220_1);  view_225 = arg220_1 = None
        convert_element_type_268: "f32[192, 512, 512]" = torch.ops.prims.convert_element_type.default(bmm_29, torch.float32);  bmm_29 = None
        view_226: "f32[8, 24, 512, 512]" = torch.ops.aten.view.default(convert_element_type_268, _shape_param_230);  convert_element_type_268 = _shape_param_230 = None
        convert_element_type_269: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(arg221_1, torch.float32);  arg221_1 = None
        mul_223: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_269, 1.1111111111111112);  convert_element_type_269 = None
        mul_224: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(view_226, mul_223);  view_226 = mul_223 = None
        convert_element_type_270: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(arg222_1, torch.float32);  arg222_1 = None
        sub_45: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_270, arg223_1);  convert_element_type_270 = arg223_1 = None
        exp_17: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_45);  sub_45 = None
        div_16: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_17, arg224_1);  exp_17 = arg224_1 = None
        mul_225: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_224, div_16);  mul_224 = None
        sum_124: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_225, [-1], True)
        neg_7: "f32[8, 24, 512, 512]" = torch.ops.aten.neg.default(div_16);  div_16 = None
        fma_7: "f32[8, 24, 512, 512]" = torch.ops.prims.fma.default(neg_7, sum_124, mul_225);  neg_7 = sum_124 = mul_225 = None
        convert_element_type_271: "bf16[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(fma_7, torch.bfloat16);  fma_7 = None
        where_10: "bf16[8, 24, 512, 512]" = torch.ops.aten.where.self(full_3, full_2, convert_element_type_271);  convert_element_type_271 = None
        view_227: "bf16[192, 512, 512]" = torch.ops.aten.view.default(where_10, _shape_param_231);  where_10 = _shape_param_231 = None
        bmm_30: "bf16[192, 64, 512]" = torch.ops.aten.bmm.default(arg225_1, view_227);  arg225_1 = None
        bmm_31: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(view_227, arg226_1);  view_227 = arg226_1 = None
        div_17: "bf16[192, 64, 512]" = torch.ops.aten.div.Tensor(bmm_30, full_4);  bmm_30 = None
        permute_83: "bf16[192, 512, 64]" = torch.ops.aten.permute.default(div_17, [0, 2, 1]);  div_17 = None
        view_228: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_28, _shape_param_232);  bmm_28 = _shape_param_232 = None
        permute_84: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_228, [0, 2, 1, 3]);  view_228 = None
        clone_29: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
        view_229: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_29, _shape_param_233);  clone_29 = _shape_param_233 = None
        view_230: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_229, _shape_param_234);  view_229 = _shape_param_234 = None
        mm_94: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_230, arg227_1);  arg227_1 = None
        permute_85: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_230, [1, 0])
        mm_95: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_85, arg228_1);  permute_85 = None
        sum_125: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_230, [0], True, dtype = torch.float32);  view_230 = None
        view_231: "f32[1536]" = torch.ops.aten.view.default(sum_125, _shape_param_235);  sum_125 = _shape_param_235 = None
        convert_element_type_272: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_231, torch.bfloat16);  view_231 = None
        view_232: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_94, _shape_param_236);  mm_94 = _shape_param_236 = None
        convert_element_type_273: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_232, torch.float32);  view_232 = None
        add_48: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_219, convert_element_type_273);  mul_219 = convert_element_type_273 = None
        convert_element_type_274: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_95, torch.float32);  mm_95 = None
        convert_element_type_275: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_272, torch.float32);  convert_element_type_272 = None
        view_233: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(permute_83, _shape_param_237);  permute_83 = _shape_param_237 = None
        permute_86: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_233, [0, 2, 1, 3]);  view_233 = None
        view_234: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(permute_86, _shape_param_238);  permute_86 = _shape_param_238 = None
        clone_30: "bf16[8, 512, 1536]" = torch.ops.aten.clone.default(view_234, memory_format = torch.contiguous_format);  view_234 = None
        view_235: "bf16[4096, 1536]" = torch.ops.aten.view.default(clone_30, _shape_param_239);  clone_30 = _shape_param_239 = None
        mm_96: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_235, arg229_1);  arg229_1 = None
        permute_87: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_235, [1, 0])
        mm_97: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_87, arg228_1);  permute_87 = None
        sum_126: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_235, [0], True, dtype = torch.float32);  view_235 = None
        view_236: "f32[1536]" = torch.ops.aten.view.default(sum_126, _shape_param_240);  sum_126 = _shape_param_240 = None
        convert_element_type_276: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_236, torch.bfloat16);  view_236 = None
        view_237: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_96, _shape_param_241);  mm_96 = _shape_param_241 = None
        convert_element_type_277: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_237, torch.float32);  view_237 = None
        add_49: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_48, convert_element_type_277);  add_48 = convert_element_type_277 = None
        convert_element_type_278: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_97, torch.float32);  mm_97 = None
        convert_element_type_279: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_276, torch.float32);  convert_element_type_276 = None
        view_238: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_31, _shape_param_242);  bmm_31 = _shape_param_242 = None
        permute_88: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_238, [0, 2, 1, 3]);  view_238 = None
        clone_31: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_88, memory_format = torch.contiguous_format);  permute_88 = None
        view_239: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_31, _shape_param_243);  clone_31 = _shape_param_243 = None
        view_240: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_239, _shape_param_244);  view_239 = _shape_param_244 = None
        mm_98: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_240, arg230_1);  arg230_1 = None
        permute_89: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_240, [1, 0])
        mm_99: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_89, arg228_1);  permute_89 = arg228_1 = None
        sum_127: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_240, [0], True, dtype = torch.float32);  view_240 = None
        view_241: "f32[1536]" = torch.ops.aten.view.default(sum_127, _shape_param_245);  sum_127 = _shape_param_245 = None
        convert_element_type_280: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_241, torch.bfloat16);  view_241 = None
        view_242: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_98, _shape_param_246);  mm_98 = _shape_param_246 = None
        convert_element_type_281: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_242, torch.float32);  view_242 = None
        add_50: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_49, convert_element_type_281);  add_49 = convert_element_type_281 = None
        convert_element_type_282: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_99, torch.float32);  mm_99 = None
        convert_element_type_283: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_280, torch.float32);  convert_element_type_280 = None
        mul_226: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_50, arg231_1);  arg231_1 = None
        mul_227: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_226, 1536)
        sum_128: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_226, [2], True)
        mul_228: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_226, arg232_1);  mul_226 = None
        sum_129: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_228, [2], True);  mul_228 = None
        mul_229: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg232_1, sum_129);  sum_129 = None
        sub_46: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_227, sum_128);  mul_227 = sum_128 = None
        sub_47: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_46, mul_229);  sub_46 = mul_229 = None
        mul_230: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg233_1, sub_47);  arg233_1 = sub_47 = None
        mul_231: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_50, arg232_1);  arg232_1 = None
        sum_130: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_231, [0, 1]);  mul_231 = None
        sum_131: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_50, [0, 1]);  add_50 = None
        convert_element_type_284: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(mul_230, torch.bfloat16)
        convert_element_type_285: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(arg234_1, torch.bfloat16);  arg234_1 = None
        mul_232: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_285, 1.1111111111111112);  convert_element_type_285 = None
        mul_233: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_284, mul_232);  convert_element_type_284 = mul_232 = None
        view_243: "bf16[4096, 1536]" = torch.ops.aten.view.default(mul_233, _shape_param_247);  mul_233 = _shape_param_247 = None
        mm_100: "bf16[4096, 6144]" = torch.ops.aten.mm.default(view_243, arg235_1);  arg235_1 = None
        permute_90: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_243, [1, 0])
        mm_101: "bf16[1536, 6144]" = torch.ops.aten.mm.default(permute_90, arg236_1);  permute_90 = arg236_1 = None
        sum_132: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_243, [0], True, dtype = torch.float32);  view_243 = None
        view_244: "f32[1536]" = torch.ops.aten.view.default(sum_132, _shape_param_248);  sum_132 = _shape_param_248 = None
        convert_element_type_286: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_244, torch.bfloat16);  view_244 = None
        view_245: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(mm_100, _shape_param_249);  mm_100 = _shape_param_249 = None
        convert_element_type_287: "f32[1536, 6144]" = torch.ops.prims.convert_element_type.default(mm_101, torch.float32);  mm_101 = None
        convert_element_type_288: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_286, torch.float32);  convert_element_type_286 = None
        convert_element_type_289: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_245, torch.float32);  view_245 = None
        view_246: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(arg237_1, _shape_param_250);  arg237_1 = _shape_param_250 = None
        convert_element_type_290: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_246, torch.float32);  view_246 = None
        mul_234: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_290, 0.7071067811865476)
        erf_9: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_234);  mul_234 = None
        add_51: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_235: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(add_51, 0.5);  add_51 = None
        mul_236: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_290, convert_element_type_290)
        mul_237: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_236, -0.5);  mul_236 = None
        exp_18: "f32[8, 512, 6144]" = torch.ops.aten.exp.default(mul_237);  mul_237 = None
        mul_238: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_18, 0.3989422804014327);  exp_18 = None
        mul_239: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_290, mul_238);  convert_element_type_290 = mul_238 = None
        add_52: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(mul_235, mul_239);  mul_235 = mul_239 = None
        mul_240: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_289, add_52);  convert_element_type_289 = add_52 = None
        convert_element_type_291: "bf16[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(mul_240, torch.bfloat16);  mul_240 = None
        view_247: "bf16[4096, 6144]" = torch.ops.aten.view.default(convert_element_type_291, _shape_param_251);  convert_element_type_291 = _shape_param_251 = None
        mm_102: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_247, arg238_1);  arg238_1 = None
        permute_91: "bf16[6144, 4096]" = torch.ops.aten.permute.default(view_247, [1, 0])
        mm_103: "bf16[6144, 1536]" = torch.ops.aten.mm.default(permute_91, arg239_1);  permute_91 = arg239_1 = None
        sum_133: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_247, [0], True, dtype = torch.float32);  view_247 = None
        view_248: "f32[6144]" = torch.ops.aten.view.default(sum_133, _shape_param_252);  sum_133 = _shape_param_252 = None
        convert_element_type_292: "bf16[6144]" = torch.ops.prims.convert_element_type.default(view_248, torch.bfloat16);  view_248 = None
        view_249: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_102, _shape_param_253);  mm_102 = _shape_param_253 = None
        convert_element_type_293: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_249, torch.float32);  view_249 = None
        add_53: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_230, convert_element_type_293);  mul_230 = convert_element_type_293 = None
        convert_element_type_294: "f32[6144, 1536]" = torch.ops.prims.convert_element_type.default(mm_103, torch.float32);  mm_103 = None
        convert_element_type_295: "f32[6144]" = torch.ops.prims.convert_element_type.default(convert_element_type_292, torch.float32);  convert_element_type_292 = None
        mul_241: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_53, arg240_1);  arg240_1 = None
        mul_242: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_241, 1536)
        sum_134: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_241, [2], True)
        mul_243: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_241, arg241_1);  mul_241 = None
        sum_135: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_243, [2], True);  mul_243 = None
        mul_244: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg241_1, sum_135);  sum_135 = None
        sub_48: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_242, sum_134);  mul_242 = sum_134 = None
        sub_49: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_48, mul_244);  sub_48 = mul_244 = None
        mul_245: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg242_1, sub_49);  arg242_1 = sub_49 = None
        mul_246: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_53, arg241_1);  arg241_1 = None
        sum_136: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_246, [0, 1]);  mul_246 = None
        sum_137: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_53, [0, 1]);  add_53 = None
        convert_element_type_296: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(mul_245, torch.bfloat16)
        convert_element_type_297: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(arg243_1, torch.bfloat16);  arg243_1 = None
        mul_247: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_297, 1.1111111111111112);  convert_element_type_297 = None
        mul_248: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_296, mul_247);  convert_element_type_296 = mul_247 = None
        view_250: "bf16[4096, 1536]" = torch.ops.aten.view.default(mul_248, _shape_param_254);  mul_248 = _shape_param_254 = None
        mm_104: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_250, arg244_1);  arg244_1 = None
        permute_92: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_250, [1, 0])
        mm_105: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_92, arg245_1);  permute_92 = arg245_1 = None
        sum_138: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_250, [0], True, dtype = torch.float32);  view_250 = None
        view_251: "f32[1536]" = torch.ops.aten.view.default(sum_138, _shape_param_255);  sum_138 = _shape_param_255 = None
        convert_element_type_298: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_251, torch.bfloat16);  view_251 = None
        view_252: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_104, _shape_param_256);  mm_104 = _shape_param_256 = None
        convert_element_type_299: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_105, torch.float32);  mm_105 = None
        convert_element_type_300: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_298, torch.float32);  convert_element_type_298 = None
        view_253: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_252, _shape_param_257);  view_252 = _shape_param_257 = None
        permute_93: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_253, [0, 2, 1, 3]);  view_253 = None
        clone_32: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_93, memory_format = torch.contiguous_format);  permute_93 = None
        view_254: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_32, _shape_param_258);  clone_32 = _shape_param_258 = None
        bmm_32: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(arg246_1, view_254);  arg246_1 = None
        bmm_33: "bf16[192, 512, 512]" = torch.ops.aten.bmm.default(view_254, arg247_1);  view_254 = arg247_1 = None
        convert_element_type_301: "f32[192, 512, 512]" = torch.ops.prims.convert_element_type.default(bmm_33, torch.float32);  bmm_33 = None
        view_255: "f32[8, 24, 512, 512]" = torch.ops.aten.view.default(convert_element_type_301, _shape_param_259);  convert_element_type_301 = _shape_param_259 = None
        convert_element_type_302: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(arg248_1, torch.float32);  arg248_1 = None
        mul_249: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_302, 1.1111111111111112);  convert_element_type_302 = None
        mul_250: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(view_255, mul_249);  view_255 = mul_249 = None
        convert_element_type_303: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(arg249_1, torch.float32);  arg249_1 = None
        sub_50: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_303, arg250_1);  convert_element_type_303 = arg250_1 = None
        exp_19: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_50);  sub_50 = None
        div_18: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_19, arg251_1);  exp_19 = arg251_1 = None
        mul_251: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_250, div_18);  mul_250 = None
        sum_139: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_251, [-1], True)
        neg_8: "f32[8, 24, 512, 512]" = torch.ops.aten.neg.default(div_18);  div_18 = None
        fma_8: "f32[8, 24, 512, 512]" = torch.ops.prims.fma.default(neg_8, sum_139, mul_251);  neg_8 = sum_139 = mul_251 = None
        convert_element_type_304: "bf16[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(fma_8, torch.bfloat16);  fma_8 = None
        where_11: "bf16[8, 24, 512, 512]" = torch.ops.aten.where.self(full_3, full_2, convert_element_type_304);  convert_element_type_304 = None
        view_256: "bf16[192, 512, 512]" = torch.ops.aten.view.default(where_11, _shape_param_260);  where_11 = _shape_param_260 = None
        bmm_34: "bf16[192, 64, 512]" = torch.ops.aten.bmm.default(arg252_1, view_256);  arg252_1 = None
        bmm_35: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(view_256, arg253_1);  view_256 = arg253_1 = None
        div_19: "bf16[192, 64, 512]" = torch.ops.aten.div.Tensor(bmm_34, full_4);  bmm_34 = None
        permute_94: "bf16[192, 512, 64]" = torch.ops.aten.permute.default(div_19, [0, 2, 1]);  div_19 = None
        view_257: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_32, _shape_param_261);  bmm_32 = _shape_param_261 = None
        permute_95: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_257, [0, 2, 1, 3]);  view_257 = None
        clone_33: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
        view_258: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_33, _shape_param_262);  clone_33 = _shape_param_262 = None
        view_259: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_258, _shape_param_263);  view_258 = _shape_param_263 = None
        mm_106: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_259, arg254_1);  arg254_1 = None
        permute_96: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_259, [1, 0])
        mm_107: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_96, arg255_1);  permute_96 = None
        sum_140: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_259, [0], True, dtype = torch.float32);  view_259 = None
        view_260: "f32[1536]" = torch.ops.aten.view.default(sum_140, _shape_param_264);  sum_140 = _shape_param_264 = None
        convert_element_type_305: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_260, torch.bfloat16);  view_260 = None
        view_261: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_106, _shape_param_265);  mm_106 = _shape_param_265 = None
        convert_element_type_306: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_261, torch.float32);  view_261 = None
        add_54: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_245, convert_element_type_306);  mul_245 = convert_element_type_306 = None
        convert_element_type_307: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_107, torch.float32);  mm_107 = None
        convert_element_type_308: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_305, torch.float32);  convert_element_type_305 = None
        view_262: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(permute_94, _shape_param_266);  permute_94 = _shape_param_266 = None
        permute_97: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_262, [0, 2, 1, 3]);  view_262 = None
        view_263: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(permute_97, _shape_param_267);  permute_97 = _shape_param_267 = None
        clone_34: "bf16[8, 512, 1536]" = torch.ops.aten.clone.default(view_263, memory_format = torch.contiguous_format);  view_263 = None
        view_264: "bf16[4096, 1536]" = torch.ops.aten.view.default(clone_34, _shape_param_268);  clone_34 = _shape_param_268 = None
        mm_108: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_264, arg256_1);  arg256_1 = None
        permute_98: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_264, [1, 0])
        mm_109: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_98, arg255_1);  permute_98 = None
        sum_141: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_264, [0], True, dtype = torch.float32);  view_264 = None
        view_265: "f32[1536]" = torch.ops.aten.view.default(sum_141, _shape_param_269);  sum_141 = _shape_param_269 = None
        convert_element_type_309: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_265, torch.bfloat16);  view_265 = None
        view_266: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_108, _shape_param_270);  mm_108 = _shape_param_270 = None
        convert_element_type_310: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_266, torch.float32);  view_266 = None
        add_55: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_54, convert_element_type_310);  add_54 = convert_element_type_310 = None
        convert_element_type_311: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_109, torch.float32);  mm_109 = None
        convert_element_type_312: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_309, torch.float32);  convert_element_type_309 = None
        view_267: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_35, _shape_param_271);  bmm_35 = _shape_param_271 = None
        permute_99: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_267, [0, 2, 1, 3]);  view_267 = None
        clone_35: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_99, memory_format = torch.contiguous_format);  permute_99 = None
        view_268: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_35, _shape_param_272);  clone_35 = _shape_param_272 = None
        view_269: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_268, _shape_param_273);  view_268 = _shape_param_273 = None
        mm_110: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_269, arg257_1);  arg257_1 = None
        permute_100: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_269, [1, 0])
        mm_111: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_100, arg255_1);  permute_100 = arg255_1 = None
        sum_142: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_269, [0], True, dtype = torch.float32);  view_269 = None
        view_270: "f32[1536]" = torch.ops.aten.view.default(sum_142, _shape_param_274);  sum_142 = _shape_param_274 = None
        convert_element_type_313: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_270, torch.bfloat16);  view_270 = None
        view_271: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_110, _shape_param_275);  mm_110 = _shape_param_275 = None
        convert_element_type_314: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_271, torch.float32);  view_271 = None
        add_56: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_55, convert_element_type_314);  add_55 = convert_element_type_314 = None
        convert_element_type_315: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_111, torch.float32);  mm_111 = None
        convert_element_type_316: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_313, torch.float32);  convert_element_type_313 = None
        mul_252: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_56, arg258_1);  arg258_1 = None
        mul_253: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_252, 1536)
        sum_143: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_252, [2], True)
        mul_254: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_252, arg259_1);  mul_252 = None
        sum_144: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_254, [2], True);  mul_254 = None
        mul_255: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg259_1, sum_144);  sum_144 = None
        sub_51: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_253, sum_143);  mul_253 = sum_143 = None
        sub_52: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_51, mul_255);  sub_51 = mul_255 = None
        mul_256: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg260_1, sub_52);  arg260_1 = sub_52 = None
        mul_257: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_56, arg259_1);  arg259_1 = None
        sum_145: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_257, [0, 1]);  mul_257 = None
        sum_146: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_56, [0, 1]);  add_56 = None
        convert_element_type_317: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(mul_256, torch.bfloat16)
        convert_element_type_318: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(arg261_1, torch.bfloat16);  arg261_1 = None
        mul_258: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_318, 1.1111111111111112);  convert_element_type_318 = None
        mul_259: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_317, mul_258);  convert_element_type_317 = mul_258 = None
        view_272: "bf16[4096, 1536]" = torch.ops.aten.view.default(mul_259, _shape_param_276);  mul_259 = _shape_param_276 = None
        mm_112: "bf16[4096, 6144]" = torch.ops.aten.mm.default(view_272, arg262_1);  arg262_1 = None
        permute_101: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_272, [1, 0])
        mm_113: "bf16[1536, 6144]" = torch.ops.aten.mm.default(permute_101, arg263_1);  permute_101 = arg263_1 = None
        sum_147: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_272, [0], True, dtype = torch.float32);  view_272 = None
        view_273: "f32[1536]" = torch.ops.aten.view.default(sum_147, _shape_param_277);  sum_147 = _shape_param_277 = None
        convert_element_type_319: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_273, torch.bfloat16);  view_273 = None
        view_274: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(mm_112, _shape_param_278);  mm_112 = _shape_param_278 = None
        convert_element_type_320: "f32[1536, 6144]" = torch.ops.prims.convert_element_type.default(mm_113, torch.float32);  mm_113 = None
        convert_element_type_321: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_319, torch.float32);  convert_element_type_319 = None
        convert_element_type_322: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_274, torch.float32);  view_274 = None
        view_275: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(arg264_1, _shape_param_279);  arg264_1 = _shape_param_279 = None
        convert_element_type_323: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_275, torch.float32);  view_275 = None
        mul_260: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_323, 0.7071067811865476)
        erf_10: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_260);  mul_260 = None
        add_57: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_261: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(add_57, 0.5);  add_57 = None
        mul_262: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_323, convert_element_type_323)
        mul_263: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_262, -0.5);  mul_262 = None
        exp_20: "f32[8, 512, 6144]" = torch.ops.aten.exp.default(mul_263);  mul_263 = None
        mul_264: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_20, 0.3989422804014327);  exp_20 = None
        mul_265: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_323, mul_264);  convert_element_type_323 = mul_264 = None
        add_58: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(mul_261, mul_265);  mul_261 = mul_265 = None
        mul_266: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_322, add_58);  convert_element_type_322 = add_58 = None
        convert_element_type_324: "bf16[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(mul_266, torch.bfloat16);  mul_266 = None
        view_276: "bf16[4096, 6144]" = torch.ops.aten.view.default(convert_element_type_324, _shape_param_280);  convert_element_type_324 = _shape_param_280 = None
        mm_114: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_276, arg265_1);  arg265_1 = None
        permute_102: "bf16[6144, 4096]" = torch.ops.aten.permute.default(view_276, [1, 0])
        mm_115: "bf16[6144, 1536]" = torch.ops.aten.mm.default(permute_102, arg266_1);  permute_102 = arg266_1 = None
        sum_148: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_276, [0], True, dtype = torch.float32);  view_276 = None
        view_277: "f32[6144]" = torch.ops.aten.view.default(sum_148, _shape_param_281);  sum_148 = _shape_param_281 = None
        convert_element_type_325: "bf16[6144]" = torch.ops.prims.convert_element_type.default(view_277, torch.bfloat16);  view_277 = None
        view_278: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_114, _shape_param_282);  mm_114 = _shape_param_282 = None
        convert_element_type_326: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_278, torch.float32);  view_278 = None
        add_59: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_256, convert_element_type_326);  mul_256 = convert_element_type_326 = None
        convert_element_type_327: "f32[6144, 1536]" = torch.ops.prims.convert_element_type.default(mm_115, torch.float32);  mm_115 = None
        convert_element_type_328: "f32[6144]" = torch.ops.prims.convert_element_type.default(convert_element_type_325, torch.float32);  convert_element_type_325 = None
        mul_267: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_59, arg267_1);  arg267_1 = None
        mul_268: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_267, 1536)
        sum_149: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_267, [2], True)
        mul_269: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_267, arg268_1);  mul_267 = None
        sum_150: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_269, [2], True);  mul_269 = None
        mul_270: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg268_1, sum_150);  sum_150 = None
        sub_53: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_268, sum_149);  mul_268 = sum_149 = None
        sub_54: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_53, mul_270);  sub_53 = mul_270 = None
        mul_271: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg269_1, sub_54);  arg269_1 = sub_54 = None
        mul_272: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_59, arg268_1);  arg268_1 = None
        sum_151: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_272, [0, 1]);  mul_272 = None
        sum_152: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_59, [0, 1]);  add_59 = None
        convert_element_type_329: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(mul_271, torch.bfloat16)
        convert_element_type_330: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(arg270_1, torch.bfloat16);  arg270_1 = None
        mul_273: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_330, 1.1111111111111112);  convert_element_type_330 = None
        mul_274: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_329, mul_273);  convert_element_type_329 = mul_273 = None
        view_279: "bf16[4096, 1536]" = torch.ops.aten.view.default(mul_274, _shape_param_283);  mul_274 = _shape_param_283 = None
        mm_116: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_279, arg271_1);  arg271_1 = None
        permute_103: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_279, [1, 0])
        mm_117: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_103, arg272_1);  permute_103 = arg272_1 = None
        sum_153: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_279, [0], True, dtype = torch.float32);  view_279 = None
        view_280: "f32[1536]" = torch.ops.aten.view.default(sum_153, _shape_param_284);  sum_153 = _shape_param_284 = None
        convert_element_type_331: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_280, torch.bfloat16);  view_280 = None
        view_281: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_116, _shape_param_285);  mm_116 = _shape_param_285 = None
        convert_element_type_332: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_117, torch.float32);  mm_117 = None
        convert_element_type_333: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_331, torch.float32);  convert_element_type_331 = None
        view_282: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_281, _shape_param_286);  view_281 = _shape_param_286 = None
        permute_104: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_282, [0, 2, 1, 3]);  view_282 = None
        clone_36: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None
        view_283: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_36, _shape_param_287);  clone_36 = _shape_param_287 = None
        bmm_36: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(arg273_1, view_283);  arg273_1 = None
        bmm_37: "bf16[192, 512, 512]" = torch.ops.aten.bmm.default(view_283, arg274_1);  view_283 = arg274_1 = None
        convert_element_type_334: "f32[192, 512, 512]" = torch.ops.prims.convert_element_type.default(bmm_37, torch.float32);  bmm_37 = None
        view_284: "f32[8, 24, 512, 512]" = torch.ops.aten.view.default(convert_element_type_334, _shape_param_288);  convert_element_type_334 = _shape_param_288 = None
        convert_element_type_335: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(arg275_1, torch.float32);  arg275_1 = None
        mul_275: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_335, 1.1111111111111112);  convert_element_type_335 = None
        mul_276: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(view_284, mul_275);  view_284 = mul_275 = None
        convert_element_type_336: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(arg276_1, torch.float32);  arg276_1 = None
        sub_55: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_336, arg277_1);  convert_element_type_336 = arg277_1 = None
        exp_21: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_55);  sub_55 = None
        div_20: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_21, arg278_1);  exp_21 = arg278_1 = None
        mul_277: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_276, div_20);  mul_276 = None
        sum_154: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_277, [-1], True)
        neg_9: "f32[8, 24, 512, 512]" = torch.ops.aten.neg.default(div_20);  div_20 = None
        fma_9: "f32[8, 24, 512, 512]" = torch.ops.prims.fma.default(neg_9, sum_154, mul_277);  neg_9 = sum_154 = mul_277 = None
        convert_element_type_337: "bf16[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(fma_9, torch.bfloat16);  fma_9 = None
        where_12: "bf16[8, 24, 512, 512]" = torch.ops.aten.where.self(full_3, full_2, convert_element_type_337);  convert_element_type_337 = None
        view_285: "bf16[192, 512, 512]" = torch.ops.aten.view.default(where_12, _shape_param_289);  where_12 = _shape_param_289 = None
        bmm_38: "bf16[192, 64, 512]" = torch.ops.aten.bmm.default(arg279_1, view_285);  arg279_1 = None
        bmm_39: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(view_285, arg280_1);  view_285 = arg280_1 = None
        div_21: "bf16[192, 64, 512]" = torch.ops.aten.div.Tensor(bmm_38, full_4);  bmm_38 = None
        permute_105: "bf16[192, 512, 64]" = torch.ops.aten.permute.default(div_21, [0, 2, 1]);  div_21 = None
        view_286: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_36, _shape_param_290);  bmm_36 = _shape_param_290 = None
        permute_106: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_286, [0, 2, 1, 3]);  view_286 = None
        clone_37: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
        view_287: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_37, _shape_param_291);  clone_37 = _shape_param_291 = None
        view_288: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_287, _shape_param_292);  view_287 = _shape_param_292 = None
        mm_118: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_288, arg281_1);  arg281_1 = None
        permute_107: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_288, [1, 0])
        mm_119: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_107, arg282_1);  permute_107 = None
        sum_155: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_288, [0], True, dtype = torch.float32);  view_288 = None
        view_289: "f32[1536]" = torch.ops.aten.view.default(sum_155, _shape_param_293);  sum_155 = _shape_param_293 = None
        convert_element_type_338: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_289, torch.bfloat16);  view_289 = None
        view_290: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_118, _shape_param_294);  mm_118 = _shape_param_294 = None
        convert_element_type_339: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_290, torch.float32);  view_290 = None
        add_60: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_271, convert_element_type_339);  mul_271 = convert_element_type_339 = None
        convert_element_type_340: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_119, torch.float32);  mm_119 = None
        convert_element_type_341: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_338, torch.float32);  convert_element_type_338 = None
        view_291: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(permute_105, _shape_param_295);  permute_105 = _shape_param_295 = None
        permute_108: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_291, [0, 2, 1, 3]);  view_291 = None
        view_292: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(permute_108, _shape_param_296);  permute_108 = _shape_param_296 = None
        clone_38: "bf16[8, 512, 1536]" = torch.ops.aten.clone.default(view_292, memory_format = torch.contiguous_format);  view_292 = None
        view_293: "bf16[4096, 1536]" = torch.ops.aten.view.default(clone_38, _shape_param_297);  clone_38 = _shape_param_297 = None
        mm_120: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_293, arg283_1);  arg283_1 = None
        permute_109: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_293, [1, 0])
        mm_121: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_109, arg282_1);  permute_109 = None
        sum_156: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_293, [0], True, dtype = torch.float32);  view_293 = None
        view_294: "f32[1536]" = torch.ops.aten.view.default(sum_156, _shape_param_298);  sum_156 = _shape_param_298 = None
        convert_element_type_342: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_294, torch.bfloat16);  view_294 = None
        view_295: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_120, _shape_param_299);  mm_120 = _shape_param_299 = None
        convert_element_type_343: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_295, torch.float32);  view_295 = None
        add_61: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_60, convert_element_type_343);  add_60 = convert_element_type_343 = None
        convert_element_type_344: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_121, torch.float32);  mm_121 = None
        convert_element_type_345: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_342, torch.float32);  convert_element_type_342 = None
        view_296: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_39, _shape_param_300);  bmm_39 = _shape_param_300 = None
        permute_110: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_296, [0, 2, 1, 3]);  view_296 = None
        clone_39: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_110, memory_format = torch.contiguous_format);  permute_110 = None
        view_297: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_39, _shape_param_301);  clone_39 = _shape_param_301 = None
        view_298: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_297, _shape_param_302);  view_297 = _shape_param_302 = None
        mm_122: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_298, arg284_1);  arg284_1 = None
        permute_111: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_298, [1, 0])
        mm_123: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_111, arg282_1);  permute_111 = arg282_1 = None
        sum_157: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_298, [0], True, dtype = torch.float32);  view_298 = None
        view_299: "f32[1536]" = torch.ops.aten.view.default(sum_157, _shape_param_303);  sum_157 = _shape_param_303 = None
        convert_element_type_346: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_299, torch.bfloat16);  view_299 = None
        view_300: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_122, _shape_param_304);  mm_122 = _shape_param_304 = None
        convert_element_type_347: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_300, torch.float32);  view_300 = None
        add_62: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_61, convert_element_type_347);  add_61 = convert_element_type_347 = None
        convert_element_type_348: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_123, torch.float32);  mm_123 = None
        convert_element_type_349: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_346, torch.float32);  convert_element_type_346 = None
        mul_278: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_62, arg285_1);  arg285_1 = None
        mul_279: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_278, 1536)
        sum_158: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_278, [2], True)
        mul_280: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_278, arg286_1);  mul_278 = None
        sum_159: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_280, [2], True);  mul_280 = None
        mul_281: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg286_1, sum_159);  sum_159 = None
        sub_56: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_279, sum_158);  mul_279 = sum_158 = None
        sub_57: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_56, mul_281);  sub_56 = mul_281 = None
        mul_282: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg287_1, sub_57);  arg287_1 = sub_57 = None
        mul_283: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_62, arg286_1);  arg286_1 = None
        sum_160: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_283, [0, 1]);  mul_283 = None
        sum_161: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_62, [0, 1]);  add_62 = None
        convert_element_type_350: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(mul_282, torch.bfloat16)
        convert_element_type_351: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(arg288_1, torch.bfloat16);  arg288_1 = None
        mul_284: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_351, 1.1111111111111112);  convert_element_type_351 = None
        mul_285: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_350, mul_284);  convert_element_type_350 = mul_284 = None
        view_301: "bf16[4096, 1536]" = torch.ops.aten.view.default(mul_285, _shape_param_305);  mul_285 = _shape_param_305 = None
        mm_124: "bf16[4096, 6144]" = torch.ops.aten.mm.default(view_301, arg289_1);  arg289_1 = None
        permute_112: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_301, [1, 0])
        mm_125: "bf16[1536, 6144]" = torch.ops.aten.mm.default(permute_112, arg290_1);  permute_112 = arg290_1 = None
        sum_162: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_301, [0], True, dtype = torch.float32);  view_301 = None
        view_302: "f32[1536]" = torch.ops.aten.view.default(sum_162, _shape_param_306);  sum_162 = _shape_param_306 = None
        convert_element_type_352: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_302, torch.bfloat16);  view_302 = None
        view_303: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(mm_124, _shape_param_307);  mm_124 = _shape_param_307 = None
        convert_element_type_353: "f32[1536, 6144]" = torch.ops.prims.convert_element_type.default(mm_125, torch.float32);  mm_125 = None
        convert_element_type_354: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_352, torch.float32);  convert_element_type_352 = None
        convert_element_type_355: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_303, torch.float32);  view_303 = None
        view_304: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(arg291_1, _shape_param_308);  arg291_1 = _shape_param_308 = None
        convert_element_type_356: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_304, torch.float32);  view_304 = None
        mul_286: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_356, 0.7071067811865476)
        erf_11: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_286);  mul_286 = None
        add_63: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_287: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(add_63, 0.5);  add_63 = None
        mul_288: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_356, convert_element_type_356)
        mul_289: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_288, -0.5);  mul_288 = None
        exp_22: "f32[8, 512, 6144]" = torch.ops.aten.exp.default(mul_289);  mul_289 = None
        mul_290: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_22, 0.3989422804014327);  exp_22 = None
        mul_291: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_356, mul_290);  convert_element_type_356 = mul_290 = None
        add_64: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(mul_287, mul_291);  mul_287 = mul_291 = None
        mul_292: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_355, add_64);  convert_element_type_355 = add_64 = None
        convert_element_type_357: "bf16[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(mul_292, torch.bfloat16);  mul_292 = None
        view_305: "bf16[4096, 6144]" = torch.ops.aten.view.default(convert_element_type_357, _shape_param_309);  convert_element_type_357 = _shape_param_309 = None
        mm_126: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_305, arg292_1);  arg292_1 = None
        permute_113: "bf16[6144, 4096]" = torch.ops.aten.permute.default(view_305, [1, 0])
        mm_127: "bf16[6144, 1536]" = torch.ops.aten.mm.default(permute_113, arg293_1);  permute_113 = arg293_1 = None
        sum_163: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_305, [0], True, dtype = torch.float32);  view_305 = None
        view_306: "f32[6144]" = torch.ops.aten.view.default(sum_163, _shape_param_310);  sum_163 = _shape_param_310 = None
        convert_element_type_358: "bf16[6144]" = torch.ops.prims.convert_element_type.default(view_306, torch.bfloat16);  view_306 = None
        view_307: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_126, _shape_param_311);  mm_126 = _shape_param_311 = None
        convert_element_type_359: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_307, torch.float32);  view_307 = None
        add_65: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_282, convert_element_type_359);  mul_282 = convert_element_type_359 = None
        convert_element_type_360: "f32[6144, 1536]" = torch.ops.prims.convert_element_type.default(mm_127, torch.float32);  mm_127 = None
        convert_element_type_361: "f32[6144]" = torch.ops.prims.convert_element_type.default(convert_element_type_358, torch.float32);  convert_element_type_358 = None
        mul_293: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_65, arg294_1);  arg294_1 = None
        mul_294: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_293, 1536)
        sum_164: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_293, [2], True)
        mul_295: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_293, arg295_1);  mul_293 = None
        sum_165: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_295, [2], True);  mul_295 = None
        mul_296: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg295_1, sum_165);  sum_165 = None
        sub_58: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_294, sum_164);  mul_294 = sum_164 = None
        sub_59: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_58, mul_296);  sub_58 = mul_296 = None
        mul_297: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg296_1, sub_59);  arg296_1 = sub_59 = None
        mul_298: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_65, arg295_1);  arg295_1 = None
        sum_166: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_298, [0, 1]);  mul_298 = None
        sum_167: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_65, [0, 1]);  add_65 = None
        convert_element_type_362: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(mul_297, torch.bfloat16)
        convert_element_type_363: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(arg297_1, torch.bfloat16);  arg297_1 = None
        mul_299: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_363, 1.1111111111111112);  convert_element_type_363 = None
        mul_300: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_362, mul_299);  convert_element_type_362 = mul_299 = None
        view_308: "bf16[4096, 1536]" = torch.ops.aten.view.default(mul_300, _shape_param_312);  mul_300 = _shape_param_312 = None
        mm_128: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_308, arg298_1);  arg298_1 = None
        permute_114: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_308, [1, 0])
        mm_129: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_114, arg299_1);  permute_114 = arg299_1 = None
        sum_168: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_308, [0], True, dtype = torch.float32);  view_308 = None
        view_309: "f32[1536]" = torch.ops.aten.view.default(sum_168, _shape_param_313);  sum_168 = _shape_param_313 = None
        convert_element_type_364: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_309, torch.bfloat16);  view_309 = None
        view_310: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_128, _shape_param_314);  mm_128 = _shape_param_314 = None
        convert_element_type_365: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_129, torch.float32);  mm_129 = None
        convert_element_type_366: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_364, torch.float32);  convert_element_type_364 = None
        view_311: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_310, _shape_param_315);  view_310 = _shape_param_315 = None
        permute_115: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_311, [0, 2, 1, 3]);  view_311 = None
        clone_40: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_115, memory_format = torch.contiguous_format);  permute_115 = None
        view_312: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_40, _shape_param_316);  clone_40 = _shape_param_316 = None
        bmm_40: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(arg300_1, view_312);  arg300_1 = None
        bmm_41: "bf16[192, 512, 512]" = torch.ops.aten.bmm.default(view_312, arg301_1);  view_312 = arg301_1 = None
        convert_element_type_367: "f32[192, 512, 512]" = torch.ops.prims.convert_element_type.default(bmm_41, torch.float32);  bmm_41 = None
        view_313: "f32[8, 24, 512, 512]" = torch.ops.aten.view.default(convert_element_type_367, _shape_param_317);  convert_element_type_367 = _shape_param_317 = None
        convert_element_type_368: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(arg302_1, torch.float32);  arg302_1 = None
        mul_301: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_368, 1.1111111111111112);  convert_element_type_368 = None
        mul_302: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(view_313, mul_301);  view_313 = mul_301 = None
        convert_element_type_369: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(arg303_1, torch.float32);  arg303_1 = None
        sub_60: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_369, arg304_1);  convert_element_type_369 = arg304_1 = None
        exp_23: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_60);  sub_60 = None
        div_22: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_23, arg305_1);  exp_23 = arg305_1 = None
        mul_303: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_302, div_22);  mul_302 = None
        sum_169: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_303, [-1], True)
        neg_10: "f32[8, 24, 512, 512]" = torch.ops.aten.neg.default(div_22);  div_22 = None
        fma_10: "f32[8, 24, 512, 512]" = torch.ops.prims.fma.default(neg_10, sum_169, mul_303);  neg_10 = sum_169 = mul_303 = None
        convert_element_type_370: "bf16[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(fma_10, torch.bfloat16);  fma_10 = None
        where_13: "bf16[8, 24, 512, 512]" = torch.ops.aten.where.self(full_3, full_2, convert_element_type_370);  convert_element_type_370 = None
        view_314: "bf16[192, 512, 512]" = torch.ops.aten.view.default(where_13, _shape_param_318);  where_13 = _shape_param_318 = None
        bmm_42: "bf16[192, 64, 512]" = torch.ops.aten.bmm.default(arg306_1, view_314);  arg306_1 = None
        bmm_43: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(view_314, arg307_1);  view_314 = arg307_1 = None
        div_23: "bf16[192, 64, 512]" = torch.ops.aten.div.Tensor(bmm_42, full_4);  bmm_42 = None
        permute_116: "bf16[192, 512, 64]" = torch.ops.aten.permute.default(div_23, [0, 2, 1]);  div_23 = None
        view_315: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_40, _shape_param_319);  bmm_40 = _shape_param_319 = None
        permute_117: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_315, [0, 2, 1, 3]);  view_315 = None
        clone_41: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
        view_316: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_41, _shape_param_320);  clone_41 = _shape_param_320 = None
        view_317: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_316, _shape_param_321);  view_316 = _shape_param_321 = None
        mm_130: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_317, arg308_1);  arg308_1 = None
        permute_118: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_317, [1, 0])
        mm_131: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_118, arg309_1);  permute_118 = None
        sum_170: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_317, [0], True, dtype = torch.float32);  view_317 = None
        view_318: "f32[1536]" = torch.ops.aten.view.default(sum_170, _shape_param_322);  sum_170 = _shape_param_322 = None
        convert_element_type_371: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_318, torch.bfloat16);  view_318 = None
        view_319: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_130, _shape_param_323);  mm_130 = _shape_param_323 = None
        convert_element_type_372: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_319, torch.float32);  view_319 = None
        add_66: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_297, convert_element_type_372);  mul_297 = convert_element_type_372 = None
        convert_element_type_373: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_131, torch.float32);  mm_131 = None
        convert_element_type_374: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_371, torch.float32);  convert_element_type_371 = None
        view_320: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(permute_116, _shape_param_324);  permute_116 = _shape_param_324 = None
        permute_119: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_320, [0, 2, 1, 3]);  view_320 = None
        view_321: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(permute_119, _shape_param_325);  permute_119 = _shape_param_325 = None
        clone_42: "bf16[8, 512, 1536]" = torch.ops.aten.clone.default(view_321, memory_format = torch.contiguous_format);  view_321 = None
        view_322: "bf16[4096, 1536]" = torch.ops.aten.view.default(clone_42, _shape_param_326);  clone_42 = _shape_param_326 = None
        mm_132: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_322, arg310_1);  arg310_1 = None
        permute_120: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_322, [1, 0])
        mm_133: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_120, arg309_1);  permute_120 = None
        sum_171: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_322, [0], True, dtype = torch.float32);  view_322 = None
        view_323: "f32[1536]" = torch.ops.aten.view.default(sum_171, _shape_param_327);  sum_171 = _shape_param_327 = None
        convert_element_type_375: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_323, torch.bfloat16);  view_323 = None
        view_324: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_132, _shape_param_328);  mm_132 = _shape_param_328 = None
        convert_element_type_376: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_324, torch.float32);  view_324 = None
        add_67: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_66, convert_element_type_376);  add_66 = convert_element_type_376 = None
        convert_element_type_377: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_133, torch.float32);  mm_133 = None
        convert_element_type_378: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_375, torch.float32);  convert_element_type_375 = None
        view_325: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_43, _shape_param_329);  bmm_43 = _shape_param_329 = None
        permute_121: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_325, [0, 2, 1, 3]);  view_325 = None
        clone_43: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_121, memory_format = torch.contiguous_format);  permute_121 = None
        view_326: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_43, _shape_param_330);  clone_43 = _shape_param_330 = None
        view_327: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_326, _shape_param_331);  view_326 = _shape_param_331 = None
        mm_134: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_327, arg311_1);  arg311_1 = None
        permute_122: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_327, [1, 0])
        mm_135: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_122, arg309_1);  permute_122 = arg309_1 = None
        sum_172: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_327, [0], True, dtype = torch.float32);  view_327 = None
        view_328: "f32[1536]" = torch.ops.aten.view.default(sum_172, _shape_param_332);  sum_172 = _shape_param_332 = None
        convert_element_type_379: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_328, torch.bfloat16);  view_328 = None
        view_329: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_134, _shape_param_333);  mm_134 = _shape_param_333 = None
        convert_element_type_380: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_329, torch.float32);  view_329 = None
        add_68: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_67, convert_element_type_380);  add_67 = convert_element_type_380 = None
        convert_element_type_381: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_135, torch.float32);  mm_135 = None
        convert_element_type_382: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_379, torch.float32);  convert_element_type_379 = None
        mul_304: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_68, arg312_1);  arg312_1 = None
        mul_305: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_304, 1536)
        sum_173: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_304, [2], True)
        mul_306: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_304, arg313_1);  mul_304 = None
        sum_174: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_306, [2], True);  mul_306 = None
        mul_307: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg313_1, sum_174);  sum_174 = None
        sub_61: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_305, sum_173);  mul_305 = sum_173 = None
        sub_62: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_61, mul_307);  sub_61 = mul_307 = None
        mul_308: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg314_1, sub_62);  arg314_1 = sub_62 = None
        mul_309: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_68, arg313_1);  arg313_1 = None
        sum_175: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_309, [0, 1]);  mul_309 = None
        sum_176: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_68, [0, 1]);  add_68 = None
        convert_element_type_383: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(mul_308, torch.bfloat16)
        convert_element_type_384: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(arg315_1, torch.bfloat16);  arg315_1 = None
        mul_310: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_384, 1.1111111111111112);  convert_element_type_384 = None
        mul_311: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_383, mul_310);  convert_element_type_383 = mul_310 = None
        view_330: "bf16[4096, 1536]" = torch.ops.aten.view.default(mul_311, _shape_param_334);  mul_311 = _shape_param_334 = None
        mm_136: "bf16[4096, 6144]" = torch.ops.aten.mm.default(view_330, arg316_1);  arg316_1 = None
        permute_123: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_330, [1, 0])
        mm_137: "bf16[1536, 6144]" = torch.ops.aten.mm.default(permute_123, arg317_1);  permute_123 = arg317_1 = None
        sum_177: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_330, [0], True, dtype = torch.float32);  view_330 = None
        view_331: "f32[1536]" = torch.ops.aten.view.default(sum_177, _shape_param_335);  sum_177 = _shape_param_335 = None
        convert_element_type_385: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_331, torch.bfloat16);  view_331 = None
        view_332: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(mm_136, _shape_param_336);  mm_136 = _shape_param_336 = None
        convert_element_type_386: "f32[1536, 6144]" = torch.ops.prims.convert_element_type.default(mm_137, torch.float32);  mm_137 = None
        convert_element_type_387: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_385, torch.float32);  convert_element_type_385 = None
        convert_element_type_388: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_332, torch.float32);  view_332 = None
        view_333: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(arg318_1, _shape_param_337);  arg318_1 = _shape_param_337 = None
        convert_element_type_389: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_333, torch.float32);  view_333 = None
        mul_312: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_389, 0.7071067811865476)
        erf_12: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_312);  mul_312 = None
        add_69: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_313: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(add_69, 0.5);  add_69 = None
        mul_314: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_389, convert_element_type_389)
        mul_315: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_314, -0.5);  mul_314 = None
        exp_24: "f32[8, 512, 6144]" = torch.ops.aten.exp.default(mul_315);  mul_315 = None
        mul_316: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_24, 0.3989422804014327);  exp_24 = None
        mul_317: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_389, mul_316);  convert_element_type_389 = mul_316 = None
        add_70: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(mul_313, mul_317);  mul_313 = mul_317 = None
        mul_318: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_388, add_70);  convert_element_type_388 = add_70 = None
        convert_element_type_390: "bf16[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(mul_318, torch.bfloat16);  mul_318 = None
        view_334: "bf16[4096, 6144]" = torch.ops.aten.view.default(convert_element_type_390, _shape_param_338);  convert_element_type_390 = _shape_param_338 = None
        mm_138: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_334, arg319_1);  arg319_1 = None
        permute_124: "bf16[6144, 4096]" = torch.ops.aten.permute.default(view_334, [1, 0])
        mm_139: "bf16[6144, 1536]" = torch.ops.aten.mm.default(permute_124, arg320_1);  permute_124 = arg320_1 = None
        sum_178: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_334, [0], True, dtype = torch.float32);  view_334 = None
        view_335: "f32[6144]" = torch.ops.aten.view.default(sum_178, _shape_param_339);  sum_178 = _shape_param_339 = None
        convert_element_type_391: "bf16[6144]" = torch.ops.prims.convert_element_type.default(view_335, torch.bfloat16);  view_335 = None
        view_336: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_138, _shape_param_340);  mm_138 = _shape_param_340 = None
        convert_element_type_392: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_336, torch.float32);  view_336 = None
        add_71: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_308, convert_element_type_392);  mul_308 = convert_element_type_392 = None
        convert_element_type_393: "f32[6144, 1536]" = torch.ops.prims.convert_element_type.default(mm_139, torch.float32);  mm_139 = None
        convert_element_type_394: "f32[6144]" = torch.ops.prims.convert_element_type.default(convert_element_type_391, torch.float32);  convert_element_type_391 = None
        mul_319: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_71, arg321_1);  arg321_1 = None
        mul_320: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_319, 1536)
        sum_179: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_319, [2], True)
        mul_321: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_319, arg322_1);  mul_319 = None
        sum_180: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_321, [2], True);  mul_321 = None
        mul_322: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg322_1, sum_180);  sum_180 = None
        sub_63: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_320, sum_179);  mul_320 = sum_179 = None
        sub_64: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_63, mul_322);  sub_63 = mul_322 = None
        mul_323: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg323_1, sub_64);  arg323_1 = sub_64 = None
        mul_324: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_71, arg322_1);  arg322_1 = None
        sum_181: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_324, [0, 1]);  mul_324 = None
        sum_182: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_71, [0, 1]);  add_71 = None
        convert_element_type_395: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(mul_323, torch.bfloat16)
        convert_element_type_396: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(arg324_1, torch.bfloat16);  arg324_1 = None
        mul_325: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_396, 1.1111111111111112);  convert_element_type_396 = None
        mul_326: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_395, mul_325);  convert_element_type_395 = mul_325 = None
        view_337: "bf16[4096, 1536]" = torch.ops.aten.view.default(mul_326, _shape_param_341);  mul_326 = _shape_param_341 = None
        mm_140: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_337, arg325_1);  arg325_1 = None
        permute_125: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_337, [1, 0])
        mm_141: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_125, arg326_1);  permute_125 = arg326_1 = None
        sum_183: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_337, [0], True, dtype = torch.float32);  view_337 = None
        view_338: "f32[1536]" = torch.ops.aten.view.default(sum_183, _shape_param_342);  sum_183 = _shape_param_342 = None
        convert_element_type_397: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_338, torch.bfloat16);  view_338 = None
        view_339: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_140, _shape_param_343);  mm_140 = _shape_param_343 = None
        convert_element_type_398: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_141, torch.float32);  mm_141 = None
        convert_element_type_399: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_397, torch.float32);  convert_element_type_397 = None
        view_340: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_339, _shape_param_344);  view_339 = _shape_param_344 = None
        permute_126: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_340, [0, 2, 1, 3]);  view_340 = None
        clone_44: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_126, memory_format = torch.contiguous_format);  permute_126 = None
        view_341: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_44, _shape_param_345);  clone_44 = _shape_param_345 = None
        bmm_44: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(arg327_1, view_341);  arg327_1 = None
        bmm_45: "bf16[192, 512, 512]" = torch.ops.aten.bmm.default(view_341, arg328_1);  view_341 = arg328_1 = None
        convert_element_type_400: "f32[192, 512, 512]" = torch.ops.prims.convert_element_type.default(bmm_45, torch.float32);  bmm_45 = None
        view_342: "f32[8, 24, 512, 512]" = torch.ops.aten.view.default(convert_element_type_400, _shape_param_346);  convert_element_type_400 = _shape_param_346 = None
        convert_element_type_401: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(arg329_1, torch.float32);  arg329_1 = None
        mul_327: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_401, 1.1111111111111112);  convert_element_type_401 = None
        mul_328: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(view_342, mul_327);  view_342 = mul_327 = None
        convert_element_type_402: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(arg330_1, torch.float32);  arg330_1 = None
        sub_65: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_402, arg331_1);  convert_element_type_402 = arg331_1 = None
        exp_25: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_65);  sub_65 = None
        div_24: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_25, arg332_1);  exp_25 = arg332_1 = None
        mul_329: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_328, div_24);  mul_328 = None
        sum_184: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_329, [-1], True)
        neg_11: "f32[8, 24, 512, 512]" = torch.ops.aten.neg.default(div_24);  div_24 = None
        fma_11: "f32[8, 24, 512, 512]" = torch.ops.prims.fma.default(neg_11, sum_184, mul_329);  neg_11 = sum_184 = mul_329 = None
        convert_element_type_403: "bf16[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(fma_11, torch.bfloat16);  fma_11 = None
        where_14: "bf16[8, 24, 512, 512]" = torch.ops.aten.where.self(full_3, full_2, convert_element_type_403);  convert_element_type_403 = None
        view_343: "bf16[192, 512, 512]" = torch.ops.aten.view.default(where_14, _shape_param_347);  where_14 = _shape_param_347 = None
        bmm_46: "bf16[192, 64, 512]" = torch.ops.aten.bmm.default(arg333_1, view_343);  arg333_1 = None
        bmm_47: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(view_343, arg334_1);  view_343 = arg334_1 = None
        div_25: "bf16[192, 64, 512]" = torch.ops.aten.div.Tensor(bmm_46, full_4);  bmm_46 = None
        permute_127: "bf16[192, 512, 64]" = torch.ops.aten.permute.default(div_25, [0, 2, 1]);  div_25 = None
        view_344: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_44, _shape_param_348);  bmm_44 = _shape_param_348 = None
        permute_128: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_344, [0, 2, 1, 3]);  view_344 = None
        clone_45: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
        view_345: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_45, _shape_param_349);  clone_45 = _shape_param_349 = None
        view_346: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_345, _shape_param_350);  view_345 = _shape_param_350 = None
        mm_142: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_346, arg335_1);  arg335_1 = None
        permute_129: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_346, [1, 0])
        mm_143: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_129, arg336_1);  permute_129 = None
        sum_185: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_346, [0], True, dtype = torch.float32);  view_346 = None
        view_347: "f32[1536]" = torch.ops.aten.view.default(sum_185, _shape_param_351);  sum_185 = _shape_param_351 = None
        convert_element_type_404: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_347, torch.bfloat16);  view_347 = None
        view_348: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_142, _shape_param_352);  mm_142 = _shape_param_352 = None
        convert_element_type_405: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_348, torch.float32);  view_348 = None
        add_72: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_323, convert_element_type_405);  mul_323 = convert_element_type_405 = None
        convert_element_type_406: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_143, torch.float32);  mm_143 = None
        convert_element_type_407: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_404, torch.float32);  convert_element_type_404 = None
        view_349: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(permute_127, _shape_param_353);  permute_127 = _shape_param_353 = None
        permute_130: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_349, [0, 2, 1, 3]);  view_349 = None
        view_350: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(permute_130, _shape_param_354);  permute_130 = _shape_param_354 = None
        clone_46: "bf16[8, 512, 1536]" = torch.ops.aten.clone.default(view_350, memory_format = torch.contiguous_format);  view_350 = None
        view_351: "bf16[4096, 1536]" = torch.ops.aten.view.default(clone_46, _shape_param_355);  clone_46 = _shape_param_355 = None
        mm_144: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_351, arg337_1);  arg337_1 = None
        permute_131: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_351, [1, 0])
        mm_145: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_131, arg336_1);  permute_131 = None
        sum_186: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_351, [0], True, dtype = torch.float32);  view_351 = None
        view_352: "f32[1536]" = torch.ops.aten.view.default(sum_186, _shape_param_356);  sum_186 = _shape_param_356 = None
        convert_element_type_408: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_352, torch.bfloat16);  view_352 = None
        view_353: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_144, _shape_param_357);  mm_144 = _shape_param_357 = None
        convert_element_type_409: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_353, torch.float32);  view_353 = None
        add_73: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_72, convert_element_type_409);  add_72 = convert_element_type_409 = None
        convert_element_type_410: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_145, torch.float32);  mm_145 = None
        convert_element_type_411: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_408, torch.float32);  convert_element_type_408 = None
        view_354: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_47, _shape_param_358);  bmm_47 = _shape_param_358 = None
        permute_132: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_354, [0, 2, 1, 3]);  view_354 = None
        clone_47: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_132, memory_format = torch.contiguous_format);  permute_132 = None
        view_355: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_47, _shape_param_359);  clone_47 = _shape_param_359 = None
        view_356: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_355, _shape_param_360);  view_355 = _shape_param_360 = None
        mm_146: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_356, arg338_1);  arg338_1 = None
        permute_133: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_356, [1, 0])
        mm_147: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_133, arg336_1);  permute_133 = arg336_1 = None
        sum_187: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_356, [0], True, dtype = torch.float32);  view_356 = None
        view_357: "f32[1536]" = torch.ops.aten.view.default(sum_187, _shape_param_361);  sum_187 = _shape_param_361 = None
        convert_element_type_412: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_357, torch.bfloat16);  view_357 = None
        view_358: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_146, _shape_param_362);  mm_146 = _shape_param_362 = None
        convert_element_type_413: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_358, torch.float32);  view_358 = None
        add_74: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_73, convert_element_type_413);  add_73 = convert_element_type_413 = None
        convert_element_type_414: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_147, torch.float32);  mm_147 = None
        convert_element_type_415: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_412, torch.float32);  convert_element_type_412 = None
        mul_330: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_74, arg339_1);  arg339_1 = None
        mul_331: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_330, 1536)
        sum_188: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_330, [2], True)
        mul_332: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_330, arg340_1);  mul_330 = None
        sum_189: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_332, [2], True);  mul_332 = None
        mul_333: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg340_1, sum_189);  sum_189 = None
        sub_66: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_331, sum_188);  mul_331 = sum_188 = None
        sub_67: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_66, mul_333);  sub_66 = mul_333 = None
        mul_334: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg341_1, sub_67);  arg341_1 = sub_67 = None
        mul_335: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_74, arg340_1);  arg340_1 = None
        sum_190: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_335, [0, 1]);  mul_335 = None
        sum_191: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_74, [0, 1]);  add_74 = None
        convert_element_type_416: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(mul_334, torch.bfloat16)
        convert_element_type_417: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(arg342_1, torch.bfloat16);  arg342_1 = None
        mul_336: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_417, 1.1111111111111112);  convert_element_type_417 = None
        mul_337: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_416, mul_336);  convert_element_type_416 = mul_336 = None
        view_359: "bf16[4096, 1536]" = torch.ops.aten.view.default(mul_337, _shape_param_363);  mul_337 = _shape_param_363 = None
        mm_148: "bf16[4096, 6144]" = torch.ops.aten.mm.default(view_359, arg343_1);  arg343_1 = None
        permute_134: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_359, [1, 0])
        mm_149: "bf16[1536, 6144]" = torch.ops.aten.mm.default(permute_134, arg344_1);  permute_134 = arg344_1 = None
        sum_192: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_359, [0], True, dtype = torch.float32);  view_359 = None
        view_360: "f32[1536]" = torch.ops.aten.view.default(sum_192, _shape_param_364);  sum_192 = _shape_param_364 = None
        convert_element_type_418: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_360, torch.bfloat16);  view_360 = None
        view_361: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(mm_148, _shape_param_365);  mm_148 = _shape_param_365 = None
        convert_element_type_419: "f32[1536, 6144]" = torch.ops.prims.convert_element_type.default(mm_149, torch.float32);  mm_149 = None
        convert_element_type_420: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_418, torch.float32);  convert_element_type_418 = None
        convert_element_type_421: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_361, torch.float32);  view_361 = None
        view_362: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(arg345_1, _shape_param_366);  arg345_1 = _shape_param_366 = None
        convert_element_type_422: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_362, torch.float32);  view_362 = None
        mul_338: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_422, 0.7071067811865476)
        erf_13: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_338);  mul_338 = None
        add_75: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
        mul_339: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(add_75, 0.5);  add_75 = None
        mul_340: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_422, convert_element_type_422)
        mul_341: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_340, -0.5);  mul_340 = None
        exp_26: "f32[8, 512, 6144]" = torch.ops.aten.exp.default(mul_341);  mul_341 = None
        mul_342: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_26, 0.3989422804014327);  exp_26 = None
        mul_343: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_422, mul_342);  convert_element_type_422 = mul_342 = None
        add_76: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(mul_339, mul_343);  mul_339 = mul_343 = None
        mul_344: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_421, add_76);  convert_element_type_421 = add_76 = None
        convert_element_type_423: "bf16[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(mul_344, torch.bfloat16);  mul_344 = None
        view_363: "bf16[4096, 6144]" = torch.ops.aten.view.default(convert_element_type_423, _shape_param_367);  convert_element_type_423 = _shape_param_367 = None
        mm_150: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_363, arg346_1);  arg346_1 = None
        permute_135: "bf16[6144, 4096]" = torch.ops.aten.permute.default(view_363, [1, 0])
        mm_151: "bf16[6144, 1536]" = torch.ops.aten.mm.default(permute_135, arg347_1);  permute_135 = arg347_1 = None
        sum_193: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_363, [0], True, dtype = torch.float32);  view_363 = None
        view_364: "f32[6144]" = torch.ops.aten.view.default(sum_193, _shape_param_368);  sum_193 = _shape_param_368 = None
        convert_element_type_424: "bf16[6144]" = torch.ops.prims.convert_element_type.default(view_364, torch.bfloat16);  view_364 = None
        view_365: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_150, _shape_param_369);  mm_150 = _shape_param_369 = None
        convert_element_type_425: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_365, torch.float32);  view_365 = None
        add_77: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_334, convert_element_type_425);  mul_334 = convert_element_type_425 = None
        convert_element_type_426: "f32[6144, 1536]" = torch.ops.prims.convert_element_type.default(mm_151, torch.float32);  mm_151 = None
        convert_element_type_427: "f32[6144]" = torch.ops.prims.convert_element_type.default(convert_element_type_424, torch.float32);  convert_element_type_424 = None
        mul_345: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_77, arg348_1);  arg348_1 = None
        mul_346: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_345, 1536)
        sum_194: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_345, [2], True)
        mul_347: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_345, arg349_1);  mul_345 = None
        sum_195: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_347, [2], True);  mul_347 = None
        mul_348: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg349_1, sum_195);  sum_195 = None
        sub_68: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_346, sum_194);  mul_346 = sum_194 = None
        sub_69: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_68, mul_348);  sub_68 = mul_348 = None
        mul_349: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg350_1, sub_69);  arg350_1 = sub_69 = None
        mul_350: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_77, arg349_1);  arg349_1 = None
        sum_196: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_350, [0, 1]);  mul_350 = None
        sum_197: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_77, [0, 1]);  add_77 = None
        convert_element_type_428: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(mul_349, torch.bfloat16)
        convert_element_type_429: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(arg351_1, torch.bfloat16);  arg351_1 = None
        mul_351: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_429, 1.1111111111111112);  convert_element_type_429 = None
        mul_352: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_428, mul_351);  convert_element_type_428 = mul_351 = None
        view_366: "bf16[4096, 1536]" = torch.ops.aten.view.default(mul_352, _shape_param_370);  mul_352 = _shape_param_370 = None
        mm_152: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_366, arg352_1);  arg352_1 = None
        permute_136: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_366, [1, 0])
        mm_153: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_136, arg353_1);  permute_136 = arg353_1 = None
        sum_198: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_366, [0], True, dtype = torch.float32);  view_366 = None
        view_367: "f32[1536]" = torch.ops.aten.view.default(sum_198, _shape_param_371);  sum_198 = _shape_param_371 = None
        convert_element_type_430: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_367, torch.bfloat16);  view_367 = None
        view_368: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_152, _shape_param_372);  mm_152 = _shape_param_372 = None
        convert_element_type_431: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_153, torch.float32);  mm_153 = None
        convert_element_type_432: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_430, torch.float32);  convert_element_type_430 = None
        view_369: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_368, _shape_param_373);  view_368 = _shape_param_373 = None
        permute_137: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_369, [0, 2, 1, 3]);  view_369 = None
        clone_48: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_137, memory_format = torch.contiguous_format);  permute_137 = None
        view_370: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_48, _shape_param_374);  clone_48 = _shape_param_374 = None
        bmm_48: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(arg354_1, view_370);  arg354_1 = None
        bmm_49: "bf16[192, 512, 512]" = torch.ops.aten.bmm.default(view_370, arg355_1);  view_370 = arg355_1 = None
        convert_element_type_433: "f32[192, 512, 512]" = torch.ops.prims.convert_element_type.default(bmm_49, torch.float32);  bmm_49 = None
        view_371: "f32[8, 24, 512, 512]" = torch.ops.aten.view.default(convert_element_type_433, _shape_param_375);  convert_element_type_433 = _shape_param_375 = None
        convert_element_type_434: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(arg356_1, torch.float32);  arg356_1 = None
        mul_353: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_434, 1.1111111111111112);  convert_element_type_434 = None
        mul_354: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(view_371, mul_353);  view_371 = mul_353 = None
        convert_element_type_435: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(arg357_1, torch.float32);  arg357_1 = None
        sub_70: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_435, arg358_1);  convert_element_type_435 = arg358_1 = None
        exp_27: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_70);  sub_70 = None
        div_26: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_27, arg359_1);  exp_27 = arg359_1 = None
        mul_355: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_354, div_26);  mul_354 = None
        sum_199: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_355, [-1], True)
        neg_12: "f32[8, 24, 512, 512]" = torch.ops.aten.neg.default(div_26);  div_26 = None
        fma_12: "f32[8, 24, 512, 512]" = torch.ops.prims.fma.default(neg_12, sum_199, mul_355);  neg_12 = sum_199 = mul_355 = None
        convert_element_type_436: "bf16[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(fma_12, torch.bfloat16);  fma_12 = None
        where_15: "bf16[8, 24, 512, 512]" = torch.ops.aten.where.self(full_3, full_2, convert_element_type_436);  convert_element_type_436 = None
        view_372: "bf16[192, 512, 512]" = torch.ops.aten.view.default(where_15, _shape_param_376);  where_15 = _shape_param_376 = None
        bmm_50: "bf16[192, 64, 512]" = torch.ops.aten.bmm.default(arg360_1, view_372);  arg360_1 = None
        bmm_51: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(view_372, arg361_1);  view_372 = arg361_1 = None
        div_27: "bf16[192, 64, 512]" = torch.ops.aten.div.Tensor(bmm_50, full_4);  bmm_50 = None
        permute_138: "bf16[192, 512, 64]" = torch.ops.aten.permute.default(div_27, [0, 2, 1]);  div_27 = None
        view_373: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_48, _shape_param_377);  bmm_48 = _shape_param_377 = None
        permute_139: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_373, [0, 2, 1, 3]);  view_373 = None
        clone_49: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_139, memory_format = torch.contiguous_format);  permute_139 = None
        view_374: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_49, _shape_param_378);  clone_49 = _shape_param_378 = None
        view_375: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_374, _shape_param_379);  view_374 = _shape_param_379 = None
        mm_154: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_375, arg362_1);  arg362_1 = None
        permute_140: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_375, [1, 0])
        mm_155: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_140, arg363_1);  permute_140 = None
        sum_200: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_375, [0], True, dtype = torch.float32);  view_375 = None
        view_376: "f32[1536]" = torch.ops.aten.view.default(sum_200, _shape_param_380);  sum_200 = _shape_param_380 = None
        convert_element_type_437: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_376, torch.bfloat16);  view_376 = None
        view_377: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_154, _shape_param_381);  mm_154 = _shape_param_381 = None
        convert_element_type_438: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_377, torch.float32);  view_377 = None
        add_78: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_349, convert_element_type_438);  mul_349 = convert_element_type_438 = None
        convert_element_type_439: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_155, torch.float32);  mm_155 = None
        convert_element_type_440: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_437, torch.float32);  convert_element_type_437 = None
        view_378: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(permute_138, _shape_param_382);  permute_138 = _shape_param_382 = None
        permute_141: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_378, [0, 2, 1, 3]);  view_378 = None
        view_379: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(permute_141, _shape_param_383);  permute_141 = _shape_param_383 = None
        clone_50: "bf16[8, 512, 1536]" = torch.ops.aten.clone.default(view_379, memory_format = torch.contiguous_format);  view_379 = None
        view_380: "bf16[4096, 1536]" = torch.ops.aten.view.default(clone_50, _shape_param_384);  clone_50 = _shape_param_384 = None
        mm_156: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_380, arg364_1);  arg364_1 = None
        permute_142: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_380, [1, 0])
        mm_157: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_142, arg363_1);  permute_142 = None
        sum_201: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_380, [0], True, dtype = torch.float32);  view_380 = None
        view_381: "f32[1536]" = torch.ops.aten.view.default(sum_201, _shape_param_385);  sum_201 = _shape_param_385 = None
        convert_element_type_441: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_381, torch.bfloat16);  view_381 = None
        view_382: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_156, _shape_param_386);  mm_156 = _shape_param_386 = None
        convert_element_type_442: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_382, torch.float32);  view_382 = None
        add_79: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_78, convert_element_type_442);  add_78 = convert_element_type_442 = None
        convert_element_type_443: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_157, torch.float32);  mm_157 = None
        convert_element_type_444: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_441, torch.float32);  convert_element_type_441 = None
        view_383: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_51, _shape_param_387);  bmm_51 = _shape_param_387 = None
        permute_143: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_383, [0, 2, 1, 3]);  view_383 = None
        clone_51: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_143, memory_format = torch.contiguous_format);  permute_143 = None
        view_384: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_51, _shape_param_388);  clone_51 = _shape_param_388 = None
        view_385: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_384, _shape_param_389);  view_384 = _shape_param_389 = None
        mm_158: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_385, arg365_1);  arg365_1 = None
        permute_144: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_385, [1, 0])
        mm_159: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_144, arg363_1);  permute_144 = arg363_1 = None
        sum_202: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_385, [0], True, dtype = torch.float32);  view_385 = None
        view_386: "f32[1536]" = torch.ops.aten.view.default(sum_202, _shape_param_390);  sum_202 = _shape_param_390 = None
        convert_element_type_445: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_386, torch.bfloat16);  view_386 = None
        view_387: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_158, _shape_param_391);  mm_158 = _shape_param_391 = None
        convert_element_type_446: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_387, torch.float32);  view_387 = None
        add_80: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_79, convert_element_type_446);  add_79 = convert_element_type_446 = None
        convert_element_type_447: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_159, torch.float32);  mm_159 = None
        convert_element_type_448: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_445, torch.float32);  convert_element_type_445 = None
        mul_356: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_80, arg366_1);  arg366_1 = None
        mul_357: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_356, 1536)
        sum_203: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_356, [2], True)
        mul_358: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_356, arg367_1);  mul_356 = None
        sum_204: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_358, [2], True);  mul_358 = None
        mul_359: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg367_1, sum_204);  sum_204 = None
        sub_71: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_357, sum_203);  mul_357 = sum_203 = None
        sub_72: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_71, mul_359);  sub_71 = mul_359 = None
        mul_360: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg368_1, sub_72);  arg368_1 = sub_72 = None
        mul_361: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_80, arg367_1);  arg367_1 = None
        sum_205: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_361, [0, 1]);  mul_361 = None
        sum_206: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_80, [0, 1]);  add_80 = None
        convert_element_type_449: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(mul_360, torch.bfloat16)
        convert_element_type_450: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(arg369_1, torch.bfloat16);  arg369_1 = None
        mul_362: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_450, 1.1111111111111112);  convert_element_type_450 = None
        mul_363: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_449, mul_362);  convert_element_type_449 = mul_362 = None
        view_388: "bf16[4096, 1536]" = torch.ops.aten.view.default(mul_363, _shape_param_392);  mul_363 = _shape_param_392 = None
        mm_160: "bf16[4096, 6144]" = torch.ops.aten.mm.default(view_388, arg370_1);  arg370_1 = None
        permute_145: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_388, [1, 0])
        mm_161: "bf16[1536, 6144]" = torch.ops.aten.mm.default(permute_145, arg371_1);  permute_145 = arg371_1 = None
        sum_207: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_388, [0], True, dtype = torch.float32);  view_388 = None
        view_389: "f32[1536]" = torch.ops.aten.view.default(sum_207, _shape_param_393);  sum_207 = _shape_param_393 = None
        convert_element_type_451: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_389, torch.bfloat16);  view_389 = None
        view_390: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(mm_160, _shape_param_394);  mm_160 = _shape_param_394 = None
        convert_element_type_452: "f32[1536, 6144]" = torch.ops.prims.convert_element_type.default(mm_161, torch.float32);  mm_161 = None
        convert_element_type_453: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_451, torch.float32);  convert_element_type_451 = None
        convert_element_type_454: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_390, torch.float32);  view_390 = None
        view_391: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(arg372_1, _shape_param_395);  arg372_1 = _shape_param_395 = None
        convert_element_type_455: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_391, torch.float32);  view_391 = None
        mul_364: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_455, 0.7071067811865476)
        erf_14: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_364);  mul_364 = None
        add_81: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
        mul_365: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(add_81, 0.5);  add_81 = None
        mul_366: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_455, convert_element_type_455)
        mul_367: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_366, -0.5);  mul_366 = None
        exp_28: "f32[8, 512, 6144]" = torch.ops.aten.exp.default(mul_367);  mul_367 = None
        mul_368: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_28, 0.3989422804014327);  exp_28 = None
        mul_369: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_455, mul_368);  convert_element_type_455 = mul_368 = None
        add_82: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(mul_365, mul_369);  mul_365 = mul_369 = None
        mul_370: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_454, add_82);  convert_element_type_454 = add_82 = None
        convert_element_type_456: "bf16[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(mul_370, torch.bfloat16);  mul_370 = None
        view_392: "bf16[4096, 6144]" = torch.ops.aten.view.default(convert_element_type_456, _shape_param_396);  convert_element_type_456 = _shape_param_396 = None
        mm_162: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_392, arg373_1);  arg373_1 = None
        permute_146: "bf16[6144, 4096]" = torch.ops.aten.permute.default(view_392, [1, 0])
        mm_163: "bf16[6144, 1536]" = torch.ops.aten.mm.default(permute_146, arg374_1);  permute_146 = arg374_1 = None
        sum_208: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_392, [0], True, dtype = torch.float32);  view_392 = None
        view_393: "f32[6144]" = torch.ops.aten.view.default(sum_208, _shape_param_397);  sum_208 = _shape_param_397 = None
        convert_element_type_457: "bf16[6144]" = torch.ops.prims.convert_element_type.default(view_393, torch.bfloat16);  view_393 = None
        view_394: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_162, _shape_param_398);  mm_162 = _shape_param_398 = None
        convert_element_type_458: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_394, torch.float32);  view_394 = None
        add_83: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_360, convert_element_type_458);  mul_360 = convert_element_type_458 = None
        convert_element_type_459: "f32[6144, 1536]" = torch.ops.prims.convert_element_type.default(mm_163, torch.float32);  mm_163 = None
        convert_element_type_460: "f32[6144]" = torch.ops.prims.convert_element_type.default(convert_element_type_457, torch.float32);  convert_element_type_457 = None
        mul_371: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_83, arg375_1);  arg375_1 = None
        mul_372: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_371, 1536)
        sum_209: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_371, [2], True)
        mul_373: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_371, arg376_1);  mul_371 = None
        sum_210: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_373, [2], True);  mul_373 = None
        mul_374: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg376_1, sum_210);  sum_210 = None
        sub_73: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_372, sum_209);  mul_372 = sum_209 = None
        sub_74: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_73, mul_374);  sub_73 = mul_374 = None
        mul_375: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg377_1, sub_74);  arg377_1 = sub_74 = None
        mul_376: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_83, arg376_1);  arg376_1 = None
        sum_211: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_376, [0, 1]);  mul_376 = None
        sum_212: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_83, [0, 1]);  add_83 = None
        convert_element_type_461: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(mul_375, torch.bfloat16)
        convert_element_type_462: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(arg378_1, torch.bfloat16);  arg378_1 = None
        mul_377: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_462, 1.1111111111111112);  convert_element_type_462 = None
        mul_378: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_461, mul_377);  convert_element_type_461 = mul_377 = None
        view_395: "bf16[4096, 1536]" = torch.ops.aten.view.default(mul_378, _shape_param_399);  mul_378 = _shape_param_399 = None
        mm_164: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_395, arg379_1);  arg379_1 = None
        permute_147: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_395, [1, 0])
        mm_165: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_147, arg380_1);  permute_147 = arg380_1 = None
        sum_213: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_395, [0], True, dtype = torch.float32);  view_395 = None
        view_396: "f32[1536]" = torch.ops.aten.view.default(sum_213, _shape_param_400);  sum_213 = _shape_param_400 = None
        convert_element_type_463: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_396, torch.bfloat16);  view_396 = None
        view_397: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_164, _shape_param_401);  mm_164 = _shape_param_401 = None
        convert_element_type_464: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_165, torch.float32);  mm_165 = None
        convert_element_type_465: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_463, torch.float32);  convert_element_type_463 = None
        view_398: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_397, _shape_param_402);  view_397 = _shape_param_402 = None
        permute_148: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_398, [0, 2, 1, 3]);  view_398 = None
        clone_52: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_148, memory_format = torch.contiguous_format);  permute_148 = None
        view_399: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_52, _shape_param_403);  clone_52 = _shape_param_403 = None
        bmm_52: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(arg381_1, view_399);  arg381_1 = None
        bmm_53: "bf16[192, 512, 512]" = torch.ops.aten.bmm.default(view_399, arg382_1);  view_399 = arg382_1 = None
        convert_element_type_466: "f32[192, 512, 512]" = torch.ops.prims.convert_element_type.default(bmm_53, torch.float32);  bmm_53 = None
        view_400: "f32[8, 24, 512, 512]" = torch.ops.aten.view.default(convert_element_type_466, _shape_param_404);  convert_element_type_466 = _shape_param_404 = None
        convert_element_type_467: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(arg383_1, torch.float32);  arg383_1 = None
        mul_379: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_467, 1.1111111111111112);  convert_element_type_467 = None
        mul_380: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(view_400, mul_379);  view_400 = mul_379 = None
        convert_element_type_468: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(arg384_1, torch.float32);  arg384_1 = None
        sub_75: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_468, arg385_1);  convert_element_type_468 = arg385_1 = None
        exp_29: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_75);  sub_75 = None
        div_28: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_29, arg386_1);  exp_29 = arg386_1 = None
        mul_381: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_380, div_28);  mul_380 = None
        sum_214: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_381, [-1], True)
        neg_13: "f32[8, 24, 512, 512]" = torch.ops.aten.neg.default(div_28);  div_28 = None
        fma_13: "f32[8, 24, 512, 512]" = torch.ops.prims.fma.default(neg_13, sum_214, mul_381);  neg_13 = sum_214 = mul_381 = None
        convert_element_type_469: "bf16[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(fma_13, torch.bfloat16);  fma_13 = None
        where_16: "bf16[8, 24, 512, 512]" = torch.ops.aten.where.self(full_3, full_2, convert_element_type_469);  convert_element_type_469 = None
        view_401: "bf16[192, 512, 512]" = torch.ops.aten.view.default(where_16, _shape_param_405);  where_16 = _shape_param_405 = None
        bmm_54: "bf16[192, 64, 512]" = torch.ops.aten.bmm.default(arg387_1, view_401);  arg387_1 = None
        bmm_55: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(view_401, arg388_1);  view_401 = arg388_1 = None
        div_29: "bf16[192, 64, 512]" = torch.ops.aten.div.Tensor(bmm_54, full_4);  bmm_54 = None
        permute_149: "bf16[192, 512, 64]" = torch.ops.aten.permute.default(div_29, [0, 2, 1]);  div_29 = None
        view_402: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_52, _shape_param_406);  bmm_52 = _shape_param_406 = None
        permute_150: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_402, [0, 2, 1, 3]);  view_402 = None
        clone_53: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_150, memory_format = torch.contiguous_format);  permute_150 = None
        view_403: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_53, _shape_param_407);  clone_53 = _shape_param_407 = None
        view_404: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_403, _shape_param_408);  view_403 = _shape_param_408 = None
        mm_166: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_404, arg389_1);  arg389_1 = None
        permute_151: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_404, [1, 0])
        mm_167: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_151, arg390_1);  permute_151 = None
        sum_215: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_404, [0], True, dtype = torch.float32);  view_404 = None
        view_405: "f32[1536]" = torch.ops.aten.view.default(sum_215, _shape_param_409);  sum_215 = _shape_param_409 = None
        convert_element_type_470: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_405, torch.bfloat16);  view_405 = None
        view_406: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_166, _shape_param_410);  mm_166 = _shape_param_410 = None
        convert_element_type_471: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_406, torch.float32);  view_406 = None
        add_84: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_375, convert_element_type_471);  mul_375 = convert_element_type_471 = None
        convert_element_type_472: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_167, torch.float32);  mm_167 = None
        convert_element_type_473: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_470, torch.float32);  convert_element_type_470 = None
        view_407: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(permute_149, _shape_param_411);  permute_149 = _shape_param_411 = None
        permute_152: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_407, [0, 2, 1, 3]);  view_407 = None
        view_408: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(permute_152, _shape_param_412);  permute_152 = _shape_param_412 = None
        clone_54: "bf16[8, 512, 1536]" = torch.ops.aten.clone.default(view_408, memory_format = torch.contiguous_format);  view_408 = None
        view_409: "bf16[4096, 1536]" = torch.ops.aten.view.default(clone_54, _shape_param_413);  clone_54 = _shape_param_413 = None
        mm_168: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_409, arg391_1);  arg391_1 = None
        permute_153: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_409, [1, 0])
        mm_169: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_153, arg390_1);  permute_153 = None
        sum_216: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_409, [0], True, dtype = torch.float32);  view_409 = None
        view_410: "f32[1536]" = torch.ops.aten.view.default(sum_216, _shape_param_414);  sum_216 = _shape_param_414 = None
        convert_element_type_474: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_410, torch.bfloat16);  view_410 = None
        view_411: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_168, _shape_param_415);  mm_168 = _shape_param_415 = None
        convert_element_type_475: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_411, torch.float32);  view_411 = None
        add_85: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_84, convert_element_type_475);  add_84 = convert_element_type_475 = None
        convert_element_type_476: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_169, torch.float32);  mm_169 = None
        convert_element_type_477: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_474, torch.float32);  convert_element_type_474 = None
        view_412: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_55, _shape_param_416);  bmm_55 = _shape_param_416 = None
        permute_154: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_412, [0, 2, 1, 3]);  view_412 = None
        clone_55: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_154, memory_format = torch.contiguous_format);  permute_154 = None
        view_413: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_55, _shape_param_417);  clone_55 = _shape_param_417 = None
        view_414: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_413, _shape_param_418);  view_413 = _shape_param_418 = None
        mm_170: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_414, arg392_1);  arg392_1 = None
        permute_155: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_414, [1, 0])
        mm_171: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_155, arg390_1);  permute_155 = arg390_1 = None
        sum_217: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_414, [0], True, dtype = torch.float32);  view_414 = None
        view_415: "f32[1536]" = torch.ops.aten.view.default(sum_217, _shape_param_419);  sum_217 = _shape_param_419 = None
        convert_element_type_478: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_415, torch.bfloat16);  view_415 = None
        view_416: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_170, _shape_param_420);  mm_170 = _shape_param_420 = None
        convert_element_type_479: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_416, torch.float32);  view_416 = None
        add_86: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_85, convert_element_type_479);  add_85 = convert_element_type_479 = None
        convert_element_type_480: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_171, torch.float32);  mm_171 = None
        convert_element_type_481: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_478, torch.float32);  convert_element_type_478 = None
        mul_382: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_86, arg393_1);  arg393_1 = None
        mul_383: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_382, 1536)
        sum_218: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_382, [2], True)
        mul_384: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_382, arg394_1);  mul_382 = None
        sum_219: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_384, [2], True);  mul_384 = None
        mul_385: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg394_1, sum_219);  sum_219 = None
        sub_76: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_383, sum_218);  mul_383 = sum_218 = None
        sub_77: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_76, mul_385);  sub_76 = mul_385 = None
        mul_386: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg395_1, sub_77);  arg395_1 = sub_77 = None
        mul_387: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_86, arg394_1);  arg394_1 = None
        sum_220: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_387, [0, 1]);  mul_387 = None
        sum_221: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_86, [0, 1]);  add_86 = None
        convert_element_type_482: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(mul_386, torch.bfloat16)
        convert_element_type_483: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(arg396_1, torch.bfloat16);  arg396_1 = None
        mul_388: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_483, 1.1111111111111112);  convert_element_type_483 = None
        mul_389: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_482, mul_388);  convert_element_type_482 = mul_388 = None
        view_417: "bf16[4096, 1536]" = torch.ops.aten.view.default(mul_389, _shape_param_421);  mul_389 = _shape_param_421 = None
        mm_172: "bf16[4096, 6144]" = torch.ops.aten.mm.default(view_417, arg397_1);  arg397_1 = None
        permute_156: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_417, [1, 0])
        mm_173: "bf16[1536, 6144]" = torch.ops.aten.mm.default(permute_156, arg398_1);  permute_156 = arg398_1 = None
        sum_222: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_417, [0], True, dtype = torch.float32);  view_417 = None
        view_418: "f32[1536]" = torch.ops.aten.view.default(sum_222, _shape_param_422);  sum_222 = _shape_param_422 = None
        convert_element_type_484: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_418, torch.bfloat16);  view_418 = None
        view_419: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(mm_172, _shape_param_423);  mm_172 = _shape_param_423 = None
        convert_element_type_485: "f32[1536, 6144]" = torch.ops.prims.convert_element_type.default(mm_173, torch.float32);  mm_173 = None
        convert_element_type_486: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_484, torch.float32);  convert_element_type_484 = None
        convert_element_type_487: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_419, torch.float32);  view_419 = None
        view_420: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(arg399_1, _shape_param_424);  arg399_1 = _shape_param_424 = None
        convert_element_type_488: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_420, torch.float32);  view_420 = None
        mul_390: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_488, 0.7071067811865476)
        erf_15: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_390);  mul_390 = None
        add_87: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
        mul_391: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(add_87, 0.5);  add_87 = None
        mul_392: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_488, convert_element_type_488)
        mul_393: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_392, -0.5);  mul_392 = None
        exp_30: "f32[8, 512, 6144]" = torch.ops.aten.exp.default(mul_393);  mul_393 = None
        mul_394: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_30, 0.3989422804014327);  exp_30 = None
        mul_395: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_488, mul_394);  convert_element_type_488 = mul_394 = None
        add_88: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(mul_391, mul_395);  mul_391 = mul_395 = None
        mul_396: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_487, add_88);  convert_element_type_487 = add_88 = None
        convert_element_type_489: "bf16[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(mul_396, torch.bfloat16);  mul_396 = None
        view_421: "bf16[4096, 6144]" = torch.ops.aten.view.default(convert_element_type_489, _shape_param_425);  convert_element_type_489 = _shape_param_425 = None
        mm_174: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_421, arg400_1);  arg400_1 = None
        permute_157: "bf16[6144, 4096]" = torch.ops.aten.permute.default(view_421, [1, 0])
        mm_175: "bf16[6144, 1536]" = torch.ops.aten.mm.default(permute_157, arg401_1);  permute_157 = arg401_1 = None
        sum_223: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_421, [0], True, dtype = torch.float32);  view_421 = None
        view_422: "f32[6144]" = torch.ops.aten.view.default(sum_223, _shape_param_426);  sum_223 = _shape_param_426 = None
        convert_element_type_490: "bf16[6144]" = torch.ops.prims.convert_element_type.default(view_422, torch.bfloat16);  view_422 = None
        view_423: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_174, _shape_param_427);  mm_174 = _shape_param_427 = None
        convert_element_type_491: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_423, torch.float32);  view_423 = None
        add_89: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_386, convert_element_type_491);  mul_386 = convert_element_type_491 = None
        convert_element_type_492: "f32[6144, 1536]" = torch.ops.prims.convert_element_type.default(mm_175, torch.float32);  mm_175 = None
        convert_element_type_493: "f32[6144]" = torch.ops.prims.convert_element_type.default(convert_element_type_490, torch.float32);  convert_element_type_490 = None
        mul_397: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_89, arg402_1);  arg402_1 = None
        mul_398: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_397, 1536)
        sum_224: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_397, [2], True)
        mul_399: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_397, arg403_1);  mul_397 = None
        sum_225: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_399, [2], True);  mul_399 = None
        mul_400: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg403_1, sum_225);  sum_225 = None
        sub_78: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_398, sum_224);  mul_398 = sum_224 = None
        sub_79: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_78, mul_400);  sub_78 = mul_400 = None
        mul_401: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg404_1, sub_79);  arg404_1 = sub_79 = None
        mul_402: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_89, arg403_1);  arg403_1 = None
        sum_226: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_402, [0, 1]);  mul_402 = None
        sum_227: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_89, [0, 1]);  add_89 = None
        convert_element_type_494: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(mul_401, torch.bfloat16)
        convert_element_type_495: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(arg405_1, torch.bfloat16);  arg405_1 = None
        mul_403: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_495, 1.1111111111111112);  convert_element_type_495 = None
        mul_404: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_494, mul_403);  convert_element_type_494 = mul_403 = None
        view_424: "bf16[4096, 1536]" = torch.ops.aten.view.default(mul_404, _shape_param_428);  mul_404 = _shape_param_428 = None
        mm_176: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_424, arg406_1);  arg406_1 = None
        permute_158: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_424, [1, 0])
        mm_177: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_158, arg407_1);  permute_158 = arg407_1 = None
        sum_228: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_424, [0], True, dtype = torch.float32);  view_424 = None
        view_425: "f32[1536]" = torch.ops.aten.view.default(sum_228, _shape_param_429);  sum_228 = _shape_param_429 = None
        convert_element_type_496: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_425, torch.bfloat16);  view_425 = None
        view_426: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_176, _shape_param_430);  mm_176 = _shape_param_430 = None
        convert_element_type_497: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_177, torch.float32);  mm_177 = None
        convert_element_type_498: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_496, torch.float32);  convert_element_type_496 = None
        view_427: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_426, _shape_param_431);  view_426 = _shape_param_431 = None
        permute_159: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_427, [0, 2, 1, 3]);  view_427 = None
        clone_56: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_159, memory_format = torch.contiguous_format);  permute_159 = None
        view_428: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_56, _shape_param_432);  clone_56 = _shape_param_432 = None
        bmm_56: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(arg408_1, view_428);  arg408_1 = None
        bmm_57: "bf16[192, 512, 512]" = torch.ops.aten.bmm.default(view_428, arg409_1);  view_428 = arg409_1 = None
        convert_element_type_499: "f32[192, 512, 512]" = torch.ops.prims.convert_element_type.default(bmm_57, torch.float32);  bmm_57 = None
        view_429: "f32[8, 24, 512, 512]" = torch.ops.aten.view.default(convert_element_type_499, _shape_param_433);  convert_element_type_499 = _shape_param_433 = None
        convert_element_type_500: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(arg410_1, torch.float32);  arg410_1 = None
        mul_405: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_500, 1.1111111111111112);  convert_element_type_500 = None
        mul_406: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(view_429, mul_405);  view_429 = mul_405 = None
        convert_element_type_501: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(arg411_1, torch.float32);  arg411_1 = None
        sub_80: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_501, arg412_1);  convert_element_type_501 = arg412_1 = None
        exp_31: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_80);  sub_80 = None
        div_30: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_31, arg413_1);  exp_31 = arg413_1 = None
        mul_407: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_406, div_30);  mul_406 = None
        sum_229: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_407, [-1], True)
        neg_14: "f32[8, 24, 512, 512]" = torch.ops.aten.neg.default(div_30);  div_30 = None
        fma_14: "f32[8, 24, 512, 512]" = torch.ops.prims.fma.default(neg_14, sum_229, mul_407);  neg_14 = sum_229 = mul_407 = None
        convert_element_type_502: "bf16[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(fma_14, torch.bfloat16);  fma_14 = None
        where_17: "bf16[8, 24, 512, 512]" = torch.ops.aten.where.self(full_3, full_2, convert_element_type_502);  convert_element_type_502 = None
        view_430: "bf16[192, 512, 512]" = torch.ops.aten.view.default(where_17, _shape_param_434);  where_17 = _shape_param_434 = None
        bmm_58: "bf16[192, 64, 512]" = torch.ops.aten.bmm.default(arg414_1, view_430);  arg414_1 = None
        bmm_59: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(view_430, arg415_1);  view_430 = arg415_1 = None
        div_31: "bf16[192, 64, 512]" = torch.ops.aten.div.Tensor(bmm_58, full_4);  bmm_58 = None
        permute_160: "bf16[192, 512, 64]" = torch.ops.aten.permute.default(div_31, [0, 2, 1]);  div_31 = None
        view_431: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_56, _shape_param_435);  bmm_56 = _shape_param_435 = None
        permute_161: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_431, [0, 2, 1, 3]);  view_431 = None
        clone_57: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_161, memory_format = torch.contiguous_format);  permute_161 = None
        view_432: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_57, _shape_param_436);  clone_57 = _shape_param_436 = None
        view_433: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_432, _shape_param_437);  view_432 = _shape_param_437 = None
        mm_178: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_433, arg416_1);  arg416_1 = None
        permute_162: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_433, [1, 0])
        mm_179: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_162, arg417_1);  permute_162 = None
        sum_230: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_433, [0], True, dtype = torch.float32);  view_433 = None
        view_434: "f32[1536]" = torch.ops.aten.view.default(sum_230, _shape_param_438);  sum_230 = _shape_param_438 = None
        convert_element_type_503: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_434, torch.bfloat16);  view_434 = None
        view_435: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_178, _shape_param_439);  mm_178 = _shape_param_439 = None
        convert_element_type_504: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_435, torch.float32);  view_435 = None
        add_90: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_401, convert_element_type_504);  mul_401 = convert_element_type_504 = None
        convert_element_type_505: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_179, torch.float32);  mm_179 = None
        convert_element_type_506: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_503, torch.float32);  convert_element_type_503 = None
        view_436: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(permute_160, _shape_param_440);  permute_160 = _shape_param_440 = None
        permute_163: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_436, [0, 2, 1, 3]);  view_436 = None
        view_437: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(permute_163, _shape_param_441);  permute_163 = _shape_param_441 = None
        clone_58: "bf16[8, 512, 1536]" = torch.ops.aten.clone.default(view_437, memory_format = torch.contiguous_format);  view_437 = None
        view_438: "bf16[4096, 1536]" = torch.ops.aten.view.default(clone_58, _shape_param_442);  clone_58 = _shape_param_442 = None
        mm_180: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_438, arg418_1);  arg418_1 = None
        permute_164: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_438, [1, 0])
        mm_181: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_164, arg417_1);  permute_164 = None
        sum_231: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_438, [0], True, dtype = torch.float32);  view_438 = None
        view_439: "f32[1536]" = torch.ops.aten.view.default(sum_231, _shape_param_443);  sum_231 = _shape_param_443 = None
        convert_element_type_507: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_439, torch.bfloat16);  view_439 = None
        view_440: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_180, _shape_param_444);  mm_180 = _shape_param_444 = None
        convert_element_type_508: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_440, torch.float32);  view_440 = None
        add_91: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_90, convert_element_type_508);  add_90 = convert_element_type_508 = None
        convert_element_type_509: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_181, torch.float32);  mm_181 = None
        convert_element_type_510: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_507, torch.float32);  convert_element_type_507 = None
        view_441: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_59, _shape_param_445);  bmm_59 = _shape_param_445 = None
        permute_165: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_441, [0, 2, 1, 3]);  view_441 = None
        clone_59: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_165, memory_format = torch.contiguous_format);  permute_165 = None
        view_442: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_59, _shape_param_446);  clone_59 = _shape_param_446 = None
        view_443: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_442, _shape_param_447);  view_442 = _shape_param_447 = None
        mm_182: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_443, arg419_1);  arg419_1 = None
        permute_166: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_443, [1, 0])
        mm_183: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_166, arg417_1);  permute_166 = arg417_1 = None
        sum_232: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_443, [0], True, dtype = torch.float32);  view_443 = None
        view_444: "f32[1536]" = torch.ops.aten.view.default(sum_232, _shape_param_448);  sum_232 = _shape_param_448 = None
        convert_element_type_511: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_444, torch.bfloat16);  view_444 = None
        view_445: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_182, _shape_param_449);  mm_182 = _shape_param_449 = None
        convert_element_type_512: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_445, torch.float32);  view_445 = None
        add_92: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_91, convert_element_type_512);  add_91 = convert_element_type_512 = None
        convert_element_type_513: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_183, torch.float32);  mm_183 = None
        convert_element_type_514: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_511, torch.float32);  convert_element_type_511 = None
        mul_408: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_92, arg420_1);  arg420_1 = None
        mul_409: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_408, 1536)
        sum_233: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_408, [2], True)
        mul_410: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_408, arg421_1);  mul_408 = None
        sum_234: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_410, [2], True);  mul_410 = None
        mul_411: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg421_1, sum_234);  sum_234 = None
        sub_81: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_409, sum_233);  mul_409 = sum_233 = None
        sub_82: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_81, mul_411);  sub_81 = mul_411 = None
        mul_412: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg422_1, sub_82);  arg422_1 = sub_82 = None
        mul_413: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_92, arg421_1);  arg421_1 = None
        sum_235: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_413, [0, 1]);  mul_413 = None
        sum_236: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_92, [0, 1]);  add_92 = None
        convert_element_type_515: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(mul_412, torch.bfloat16)
        convert_element_type_516: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(arg423_1, torch.bfloat16);  arg423_1 = None
        mul_414: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_516, 1.1111111111111112);  convert_element_type_516 = None
        mul_415: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_515, mul_414);  convert_element_type_515 = mul_414 = None
        view_446: "bf16[4096, 1536]" = torch.ops.aten.view.default(mul_415, _shape_param_450);  mul_415 = _shape_param_450 = None
        mm_184: "bf16[4096, 6144]" = torch.ops.aten.mm.default(view_446, arg424_1);  arg424_1 = None
        permute_167: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_446, [1, 0])
        mm_185: "bf16[1536, 6144]" = torch.ops.aten.mm.default(permute_167, arg425_1);  permute_167 = arg425_1 = None
        sum_237: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_446, [0], True, dtype = torch.float32);  view_446 = None
        view_447: "f32[1536]" = torch.ops.aten.view.default(sum_237, _shape_param_451);  sum_237 = _shape_param_451 = None
        convert_element_type_517: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_447, torch.bfloat16);  view_447 = None
        view_448: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(mm_184, _shape_param_452);  mm_184 = _shape_param_452 = None
        convert_element_type_518: "f32[1536, 6144]" = torch.ops.prims.convert_element_type.default(mm_185, torch.float32);  mm_185 = None
        convert_element_type_519: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_517, torch.float32);  convert_element_type_517 = None
        convert_element_type_520: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_448, torch.float32);  view_448 = None
        view_449: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(arg426_1, _shape_param_453);  arg426_1 = _shape_param_453 = None
        convert_element_type_521: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_449, torch.float32);  view_449 = None
        mul_416: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_521, 0.7071067811865476)
        erf_16: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_416);  mul_416 = None
        add_93: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
        mul_417: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(add_93, 0.5);  add_93 = None
        mul_418: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_521, convert_element_type_521)
        mul_419: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_418, -0.5);  mul_418 = None
        exp_32: "f32[8, 512, 6144]" = torch.ops.aten.exp.default(mul_419);  mul_419 = None
        mul_420: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_32, 0.3989422804014327);  exp_32 = None
        mul_421: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_521, mul_420);  convert_element_type_521 = mul_420 = None
        add_94: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(mul_417, mul_421);  mul_417 = mul_421 = None
        mul_422: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_520, add_94);  convert_element_type_520 = add_94 = None
        convert_element_type_522: "bf16[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(mul_422, torch.bfloat16);  mul_422 = None
        view_450: "bf16[4096, 6144]" = torch.ops.aten.view.default(convert_element_type_522, _shape_param_454);  convert_element_type_522 = _shape_param_454 = None
        mm_186: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_450, arg427_1);  arg427_1 = None
        permute_168: "bf16[6144, 4096]" = torch.ops.aten.permute.default(view_450, [1, 0])
        mm_187: "bf16[6144, 1536]" = torch.ops.aten.mm.default(permute_168, arg428_1);  permute_168 = arg428_1 = None
        sum_238: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_450, [0], True, dtype = torch.float32);  view_450 = None
        view_451: "f32[6144]" = torch.ops.aten.view.default(sum_238, _shape_param_455);  sum_238 = _shape_param_455 = None
        convert_element_type_523: "bf16[6144]" = torch.ops.prims.convert_element_type.default(view_451, torch.bfloat16);  view_451 = None
        view_452: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_186, _shape_param_456);  mm_186 = _shape_param_456 = None
        convert_element_type_524: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_452, torch.float32);  view_452 = None
        add_95: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_412, convert_element_type_524);  mul_412 = convert_element_type_524 = None
        convert_element_type_525: "f32[6144, 1536]" = torch.ops.prims.convert_element_type.default(mm_187, torch.float32);  mm_187 = None
        convert_element_type_526: "f32[6144]" = torch.ops.prims.convert_element_type.default(convert_element_type_523, torch.float32);  convert_element_type_523 = None
        mul_423: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_95, arg429_1);  arg429_1 = None
        mul_424: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_423, 1536)
        sum_239: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_423, [2], True)
        mul_425: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_423, arg430_1);  mul_423 = None
        sum_240: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_425, [2], True);  mul_425 = None
        mul_426: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg430_1, sum_240);  sum_240 = None
        sub_83: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_424, sum_239);  mul_424 = sum_239 = None
        sub_84: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_83, mul_426);  sub_83 = mul_426 = None
        mul_427: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg431_1, sub_84);  arg431_1 = sub_84 = None
        mul_428: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_95, arg430_1);  arg430_1 = None
        sum_241: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_428, [0, 1]);  mul_428 = None
        sum_242: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_95, [0, 1]);  add_95 = None
        convert_element_type_527: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(mul_427, torch.bfloat16)
        convert_element_type_528: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(arg432_1, torch.bfloat16);  arg432_1 = None
        mul_429: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_528, 1.1111111111111112);  convert_element_type_528 = None
        mul_430: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_527, mul_429);  convert_element_type_527 = mul_429 = None
        view_453: "bf16[4096, 1536]" = torch.ops.aten.view.default(mul_430, _shape_param_457);  mul_430 = _shape_param_457 = None
        mm_188: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_453, arg433_1);  arg433_1 = None
        permute_169: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_453, [1, 0])
        mm_189: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_169, arg434_1);  permute_169 = arg434_1 = None
        sum_243: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_453, [0], True, dtype = torch.float32);  view_453 = None
        view_454: "f32[1536]" = torch.ops.aten.view.default(sum_243, _shape_param_458);  sum_243 = _shape_param_458 = None
        convert_element_type_529: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_454, torch.bfloat16);  view_454 = None
        view_455: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_188, _shape_param_459);  mm_188 = _shape_param_459 = None
        convert_element_type_530: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_189, torch.float32);  mm_189 = None
        convert_element_type_531: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_529, torch.float32);  convert_element_type_529 = None
        view_456: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_455, _shape_param_460);  view_455 = _shape_param_460 = None
        permute_170: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_456, [0, 2, 1, 3]);  view_456 = None
        clone_60: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_170, memory_format = torch.contiguous_format);  permute_170 = None
        view_457: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_60, _shape_param_461);  clone_60 = _shape_param_461 = None
        bmm_60: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(arg435_1, view_457);  arg435_1 = None
        bmm_61: "bf16[192, 512, 512]" = torch.ops.aten.bmm.default(view_457, arg436_1);  view_457 = arg436_1 = None
        convert_element_type_532: "f32[192, 512, 512]" = torch.ops.prims.convert_element_type.default(bmm_61, torch.float32);  bmm_61 = None
        view_458: "f32[8, 24, 512, 512]" = torch.ops.aten.view.default(convert_element_type_532, _shape_param_462);  convert_element_type_532 = _shape_param_462 = None
        convert_element_type_533: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(arg437_1, torch.float32);  arg437_1 = None
        mul_431: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_533, 1.1111111111111112);  convert_element_type_533 = None
        mul_432: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(view_458, mul_431);  view_458 = mul_431 = None
        convert_element_type_534: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(arg438_1, torch.float32);  arg438_1 = None
        sub_85: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_534, arg439_1);  convert_element_type_534 = arg439_1 = None
        exp_33: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_85);  sub_85 = None
        div_32: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_33, arg440_1);  exp_33 = arg440_1 = None
        mul_433: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_432, div_32);  mul_432 = None
        sum_244: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_433, [-1], True)
        neg_15: "f32[8, 24, 512, 512]" = torch.ops.aten.neg.default(div_32);  div_32 = None
        fma_15: "f32[8, 24, 512, 512]" = torch.ops.prims.fma.default(neg_15, sum_244, mul_433);  neg_15 = sum_244 = mul_433 = None
        convert_element_type_535: "bf16[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(fma_15, torch.bfloat16);  fma_15 = None
        where_18: "bf16[8, 24, 512, 512]" = torch.ops.aten.where.self(full_3, full_2, convert_element_type_535);  convert_element_type_535 = None
        view_459: "bf16[192, 512, 512]" = torch.ops.aten.view.default(where_18, _shape_param_463);  where_18 = _shape_param_463 = None
        bmm_62: "bf16[192, 64, 512]" = torch.ops.aten.bmm.default(arg441_1, view_459);  arg441_1 = None
        bmm_63: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(view_459, arg442_1);  view_459 = arg442_1 = None
        div_33: "bf16[192, 64, 512]" = torch.ops.aten.div.Tensor(bmm_62, full_4);  bmm_62 = None
        permute_171: "bf16[192, 512, 64]" = torch.ops.aten.permute.default(div_33, [0, 2, 1]);  div_33 = None
        view_460: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_60, _shape_param_464);  bmm_60 = _shape_param_464 = None
        permute_172: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_460, [0, 2, 1, 3]);  view_460 = None
        clone_61: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_172, memory_format = torch.contiguous_format);  permute_172 = None
        view_461: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_61, _shape_param_465);  clone_61 = _shape_param_465 = None
        view_462: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_461, _shape_param_466);  view_461 = _shape_param_466 = None
        mm_190: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_462, arg443_1);  arg443_1 = None
        permute_173: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_462, [1, 0])
        mm_191: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_173, arg444_1);  permute_173 = None
        sum_245: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_462, [0], True, dtype = torch.float32);  view_462 = None
        view_463: "f32[1536]" = torch.ops.aten.view.default(sum_245, _shape_param_467);  sum_245 = _shape_param_467 = None
        convert_element_type_536: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_463, torch.bfloat16);  view_463 = None
        view_464: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_190, _shape_param_468);  mm_190 = _shape_param_468 = None
        convert_element_type_537: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_464, torch.float32);  view_464 = None
        add_96: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_427, convert_element_type_537);  mul_427 = convert_element_type_537 = None
        convert_element_type_538: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_191, torch.float32);  mm_191 = None
        convert_element_type_539: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_536, torch.float32);  convert_element_type_536 = None
        view_465: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(permute_171, _shape_param_469);  permute_171 = _shape_param_469 = None
        permute_174: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_465, [0, 2, 1, 3]);  view_465 = None
        view_466: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(permute_174, _shape_param_470);  permute_174 = _shape_param_470 = None
        clone_62: "bf16[8, 512, 1536]" = torch.ops.aten.clone.default(view_466, memory_format = torch.contiguous_format);  view_466 = None
        view_467: "bf16[4096, 1536]" = torch.ops.aten.view.default(clone_62, _shape_param_471);  clone_62 = _shape_param_471 = None
        mm_192: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_467, arg445_1);  arg445_1 = None
        permute_175: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_467, [1, 0])
        mm_193: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_175, arg444_1);  permute_175 = None
        sum_246: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_467, [0], True, dtype = torch.float32);  view_467 = None
        view_468: "f32[1536]" = torch.ops.aten.view.default(sum_246, _shape_param_472);  sum_246 = _shape_param_472 = None
        convert_element_type_540: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_468, torch.bfloat16);  view_468 = None
        view_469: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_192, _shape_param_473);  mm_192 = _shape_param_473 = None
        convert_element_type_541: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_469, torch.float32);  view_469 = None
        add_97: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_96, convert_element_type_541);  add_96 = convert_element_type_541 = None
        convert_element_type_542: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_193, torch.float32);  mm_193 = None
        convert_element_type_543: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_540, torch.float32);  convert_element_type_540 = None
        view_470: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_63, _shape_param_474);  bmm_63 = _shape_param_474 = None
        permute_176: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_470, [0, 2, 1, 3]);  view_470 = None
        clone_63: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_176, memory_format = torch.contiguous_format);  permute_176 = None
        view_471: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_63, _shape_param_475);  clone_63 = _shape_param_475 = None
        view_472: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_471, _shape_param_476);  view_471 = _shape_param_476 = None
        mm_194: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_472, arg446_1);  arg446_1 = None
        permute_177: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_472, [1, 0])
        mm_195: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_177, arg444_1);  permute_177 = arg444_1 = None
        sum_247: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_472, [0], True, dtype = torch.float32);  view_472 = None
        view_473: "f32[1536]" = torch.ops.aten.view.default(sum_247, _shape_param_477);  sum_247 = _shape_param_477 = None
        convert_element_type_544: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_473, torch.bfloat16);  view_473 = None
        view_474: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_194, _shape_param_478);  mm_194 = _shape_param_478 = None
        convert_element_type_545: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_474, torch.float32);  view_474 = None
        add_98: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_97, convert_element_type_545);  add_97 = convert_element_type_545 = None
        convert_element_type_546: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_195, torch.float32);  mm_195 = None
        convert_element_type_547: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_544, torch.float32);  convert_element_type_544 = None
        mul_434: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_98, arg447_1);  arg447_1 = None
        mul_435: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_434, 1536)
        sum_248: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_434, [2], True)
        mul_436: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_434, arg448_1);  mul_434 = None
        sum_249: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_436, [2], True);  mul_436 = None
        mul_437: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg448_1, sum_249);  sum_249 = None
        sub_86: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_435, sum_248);  mul_435 = sum_248 = None
        sub_87: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_86, mul_437);  sub_86 = mul_437 = None
        mul_438: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg449_1, sub_87);  arg449_1 = sub_87 = None
        mul_439: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_98, arg448_1);  arg448_1 = None
        sum_250: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_439, [0, 1]);  mul_439 = None
        sum_251: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_98, [0, 1]);  add_98 = None
        convert_element_type_548: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(mul_438, torch.bfloat16)
        convert_element_type_549: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(arg450_1, torch.bfloat16);  arg450_1 = None
        mul_440: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_549, 1.1111111111111112);  convert_element_type_549 = None
        mul_441: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_548, mul_440);  convert_element_type_548 = mul_440 = None
        view_475: "bf16[4096, 1536]" = torch.ops.aten.view.default(mul_441, _shape_param_479);  mul_441 = _shape_param_479 = None
        mm_196: "bf16[4096, 6144]" = torch.ops.aten.mm.default(view_475, arg451_1);  arg451_1 = None
        permute_178: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_475, [1, 0])
        mm_197: "bf16[1536, 6144]" = torch.ops.aten.mm.default(permute_178, arg452_1);  permute_178 = arg452_1 = None
        sum_252: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_475, [0], True, dtype = torch.float32);  view_475 = None
        view_476: "f32[1536]" = torch.ops.aten.view.default(sum_252, _shape_param_480);  sum_252 = _shape_param_480 = None
        convert_element_type_550: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_476, torch.bfloat16);  view_476 = None
        view_477: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(mm_196, _shape_param_481);  mm_196 = _shape_param_481 = None
        convert_element_type_551: "f32[1536, 6144]" = torch.ops.prims.convert_element_type.default(mm_197, torch.float32);  mm_197 = None
        convert_element_type_552: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_550, torch.float32);  convert_element_type_550 = None
        convert_element_type_553: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_477, torch.float32);  view_477 = None
        view_478: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(arg453_1, _shape_param_482);  arg453_1 = _shape_param_482 = None
        convert_element_type_554: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_478, torch.float32);  view_478 = None
        mul_442: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_554, 0.7071067811865476)
        erf_17: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_442);  mul_442 = None
        add_99: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
        mul_443: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(add_99, 0.5);  add_99 = None
        mul_444: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_554, convert_element_type_554)
        mul_445: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_444, -0.5);  mul_444 = None
        exp_34: "f32[8, 512, 6144]" = torch.ops.aten.exp.default(mul_445);  mul_445 = None
        mul_446: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_34, 0.3989422804014327);  exp_34 = None
        mul_447: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_554, mul_446);  convert_element_type_554 = mul_446 = None
        add_100: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(mul_443, mul_447);  mul_443 = mul_447 = None
        mul_448: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_553, add_100);  convert_element_type_553 = add_100 = None
        convert_element_type_555: "bf16[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(mul_448, torch.bfloat16);  mul_448 = None
        view_479: "bf16[4096, 6144]" = torch.ops.aten.view.default(convert_element_type_555, _shape_param_483);  convert_element_type_555 = _shape_param_483 = None
        mm_198: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_479, arg454_1);  arg454_1 = None
        permute_179: "bf16[6144, 4096]" = torch.ops.aten.permute.default(view_479, [1, 0])
        mm_199: "bf16[6144, 1536]" = torch.ops.aten.mm.default(permute_179, arg455_1);  permute_179 = arg455_1 = None
        sum_253: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_479, [0], True, dtype = torch.float32);  view_479 = None
        view_480: "f32[6144]" = torch.ops.aten.view.default(sum_253, _shape_param_484);  sum_253 = _shape_param_484 = None
        convert_element_type_556: "bf16[6144]" = torch.ops.prims.convert_element_type.default(view_480, torch.bfloat16);  view_480 = None
        view_481: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_198, _shape_param_485);  mm_198 = _shape_param_485 = None
        convert_element_type_557: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_481, torch.float32);  view_481 = None
        add_101: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_438, convert_element_type_557);  mul_438 = convert_element_type_557 = None
        convert_element_type_558: "f32[6144, 1536]" = torch.ops.prims.convert_element_type.default(mm_199, torch.float32);  mm_199 = None
        convert_element_type_559: "f32[6144]" = torch.ops.prims.convert_element_type.default(convert_element_type_556, torch.float32);  convert_element_type_556 = None
        mul_449: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_101, arg456_1);  arg456_1 = None
        mul_450: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_449, 1536)
        sum_254: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_449, [2], True)
        mul_451: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_449, arg457_1);  mul_449 = None
        sum_255: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_451, [2], True);  mul_451 = None
        mul_452: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg457_1, sum_255);  sum_255 = None
        sub_88: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_450, sum_254);  mul_450 = sum_254 = None
        sub_89: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_88, mul_452);  sub_88 = mul_452 = None
        mul_453: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg458_1, sub_89);  arg458_1 = sub_89 = None
        mul_454: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_101, arg457_1);  arg457_1 = None
        sum_256: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_454, [0, 1]);  mul_454 = None
        sum_257: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_101, [0, 1]);  add_101 = None
        convert_element_type_560: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(mul_453, torch.bfloat16)
        convert_element_type_561: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(arg459_1, torch.bfloat16);  arg459_1 = None
        mul_455: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_561, 1.1111111111111112);  convert_element_type_561 = None
        mul_456: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_560, mul_455);  convert_element_type_560 = mul_455 = None
        view_482: "bf16[4096, 1536]" = torch.ops.aten.view.default(mul_456, _shape_param_486);  mul_456 = _shape_param_486 = None
        mm_200: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_482, arg460_1);  arg460_1 = None
        permute_180: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_482, [1, 0])
        mm_201: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_180, arg461_1);  permute_180 = arg461_1 = None
        sum_258: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_482, [0], True, dtype = torch.float32);  view_482 = None
        view_483: "f32[1536]" = torch.ops.aten.view.default(sum_258, _shape_param_487);  sum_258 = _shape_param_487 = None
        convert_element_type_562: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_483, torch.bfloat16);  view_483 = None
        view_484: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_200, _shape_param_488);  mm_200 = _shape_param_488 = None
        convert_element_type_563: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_201, torch.float32);  mm_201 = None
        convert_element_type_564: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_562, torch.float32);  convert_element_type_562 = None
        view_485: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_484, _shape_param_489);  view_484 = _shape_param_489 = None
        permute_181: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_485, [0, 2, 1, 3]);  view_485 = None
        clone_64: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_181, memory_format = torch.contiguous_format);  permute_181 = None
        view_486: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_64, _shape_param_490);  clone_64 = _shape_param_490 = None
        bmm_64: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(arg462_1, view_486);  arg462_1 = None
        bmm_65: "bf16[192, 512, 512]" = torch.ops.aten.bmm.default(view_486, arg463_1);  view_486 = arg463_1 = None
        convert_element_type_565: "f32[192, 512, 512]" = torch.ops.prims.convert_element_type.default(bmm_65, torch.float32);  bmm_65 = None
        view_487: "f32[8, 24, 512, 512]" = torch.ops.aten.view.default(convert_element_type_565, _shape_param_491);  convert_element_type_565 = _shape_param_491 = None
        convert_element_type_566: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(arg464_1, torch.float32);  arg464_1 = None
        mul_457: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_566, 1.1111111111111112);  convert_element_type_566 = None
        mul_458: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(view_487, mul_457);  view_487 = mul_457 = None
        convert_element_type_567: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(arg465_1, torch.float32);  arg465_1 = None
        sub_90: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_567, arg466_1);  convert_element_type_567 = arg466_1 = None
        exp_35: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_90);  sub_90 = None
        div_34: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_35, arg467_1);  exp_35 = arg467_1 = None
        mul_459: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_458, div_34);  mul_458 = None
        sum_259: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_459, [-1], True)
        neg_16: "f32[8, 24, 512, 512]" = torch.ops.aten.neg.default(div_34);  div_34 = None
        fma_16: "f32[8, 24, 512, 512]" = torch.ops.prims.fma.default(neg_16, sum_259, mul_459);  neg_16 = sum_259 = mul_459 = None
        convert_element_type_568: "bf16[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(fma_16, torch.bfloat16);  fma_16 = None
        where_19: "bf16[8, 24, 512, 512]" = torch.ops.aten.where.self(full_3, full_2, convert_element_type_568);  convert_element_type_568 = None
        view_488: "bf16[192, 512, 512]" = torch.ops.aten.view.default(where_19, _shape_param_492);  where_19 = _shape_param_492 = None
        bmm_66: "bf16[192, 64, 512]" = torch.ops.aten.bmm.default(arg468_1, view_488);  arg468_1 = None
        bmm_67: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(view_488, arg469_1);  view_488 = arg469_1 = None
        div_35: "bf16[192, 64, 512]" = torch.ops.aten.div.Tensor(bmm_66, full_4);  bmm_66 = None
        permute_182: "bf16[192, 512, 64]" = torch.ops.aten.permute.default(div_35, [0, 2, 1]);  div_35 = None
        view_489: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_64, _shape_param_493);  bmm_64 = _shape_param_493 = None
        permute_183: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_489, [0, 2, 1, 3]);  view_489 = None
        clone_65: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_183, memory_format = torch.contiguous_format);  permute_183 = None
        view_490: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_65, _shape_param_494);  clone_65 = _shape_param_494 = None
        view_491: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_490, _shape_param_495);  view_490 = _shape_param_495 = None
        mm_202: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_491, arg470_1);  arg470_1 = None
        permute_184: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_491, [1, 0])
        mm_203: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_184, arg471_1);  permute_184 = None
        sum_260: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_491, [0], True, dtype = torch.float32);  view_491 = None
        view_492: "f32[1536]" = torch.ops.aten.view.default(sum_260, _shape_param_496);  sum_260 = _shape_param_496 = None
        convert_element_type_569: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_492, torch.bfloat16);  view_492 = None
        view_493: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_202, _shape_param_497);  mm_202 = _shape_param_497 = None
        convert_element_type_570: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_493, torch.float32);  view_493 = None
        add_102: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_453, convert_element_type_570);  mul_453 = convert_element_type_570 = None
        convert_element_type_571: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_203, torch.float32);  mm_203 = None
        convert_element_type_572: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_569, torch.float32);  convert_element_type_569 = None
        view_494: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(permute_182, _shape_param_498);  permute_182 = _shape_param_498 = None
        permute_185: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_494, [0, 2, 1, 3]);  view_494 = None
        view_495: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(permute_185, _shape_param_499);  permute_185 = _shape_param_499 = None
        clone_66: "bf16[8, 512, 1536]" = torch.ops.aten.clone.default(view_495, memory_format = torch.contiguous_format);  view_495 = None
        view_496: "bf16[4096, 1536]" = torch.ops.aten.view.default(clone_66, _shape_param_500);  clone_66 = _shape_param_500 = None
        mm_204: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_496, arg472_1);  arg472_1 = None
        permute_186: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_496, [1, 0])
        mm_205: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_186, arg471_1);  permute_186 = None
        sum_261: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_496, [0], True, dtype = torch.float32);  view_496 = None
        view_497: "f32[1536]" = torch.ops.aten.view.default(sum_261, _shape_param_501);  sum_261 = _shape_param_501 = None
        convert_element_type_573: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_497, torch.bfloat16);  view_497 = None
        view_498: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_204, _shape_param_502);  mm_204 = _shape_param_502 = None
        convert_element_type_574: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_498, torch.float32);  view_498 = None
        add_103: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_102, convert_element_type_574);  add_102 = convert_element_type_574 = None
        convert_element_type_575: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_205, torch.float32);  mm_205 = None
        convert_element_type_576: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_573, torch.float32);  convert_element_type_573 = None
        view_499: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_67, _shape_param_503);  bmm_67 = _shape_param_503 = None
        permute_187: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_499, [0, 2, 1, 3]);  view_499 = None
        clone_67: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_187, memory_format = torch.contiguous_format);  permute_187 = None
        view_500: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_67, _shape_param_504);  clone_67 = _shape_param_504 = None
        view_501: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_500, _shape_param_505);  view_500 = _shape_param_505 = None
        mm_206: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_501, arg473_1);  arg473_1 = None
        permute_188: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_501, [1, 0])
        mm_207: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_188, arg471_1);  permute_188 = arg471_1 = None
        sum_262: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_501, [0], True, dtype = torch.float32);  view_501 = None
        view_502: "f32[1536]" = torch.ops.aten.view.default(sum_262, _shape_param_506);  sum_262 = _shape_param_506 = None
        convert_element_type_577: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_502, torch.bfloat16);  view_502 = None
        view_503: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_206, _shape_param_507);  mm_206 = _shape_param_507 = None
        convert_element_type_578: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_503, torch.float32);  view_503 = None
        add_104: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_103, convert_element_type_578);  add_103 = convert_element_type_578 = None
        convert_element_type_579: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_207, torch.float32);  mm_207 = None
        convert_element_type_580: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_577, torch.float32);  convert_element_type_577 = None
        mul_460: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_104, arg474_1);  arg474_1 = None
        mul_461: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_460, 1536)
        sum_263: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_460, [2], True)
        mul_462: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_460, arg475_1);  mul_460 = None
        sum_264: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_462, [2], True);  mul_462 = None
        mul_463: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg475_1, sum_264);  sum_264 = None
        sub_91: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_461, sum_263);  mul_461 = sum_263 = None
        sub_92: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_91, mul_463);  sub_91 = mul_463 = None
        mul_464: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg476_1, sub_92);  arg476_1 = sub_92 = None
        mul_465: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_104, arg475_1);  arg475_1 = None
        sum_265: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_465, [0, 1]);  mul_465 = None
        sum_266: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_104, [0, 1]);  add_104 = None
        convert_element_type_581: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(mul_464, torch.bfloat16)
        convert_element_type_582: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(arg477_1, torch.bfloat16);  arg477_1 = None
        mul_466: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_582, 1.1111111111111112);  convert_element_type_582 = None
        mul_467: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_581, mul_466);  convert_element_type_581 = mul_466 = None
        view_504: "bf16[4096, 1536]" = torch.ops.aten.view.default(mul_467, _shape_param_508);  mul_467 = _shape_param_508 = None
        mm_208: "bf16[4096, 6144]" = torch.ops.aten.mm.default(view_504, arg478_1);  arg478_1 = None
        permute_189: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_504, [1, 0])
        mm_209: "bf16[1536, 6144]" = torch.ops.aten.mm.default(permute_189, arg479_1);  permute_189 = arg479_1 = None
        sum_267: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_504, [0], True, dtype = torch.float32);  view_504 = None
        view_505: "f32[1536]" = torch.ops.aten.view.default(sum_267, _shape_param_509);  sum_267 = _shape_param_509 = None
        convert_element_type_583: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_505, torch.bfloat16);  view_505 = None
        view_506: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(mm_208, _shape_param_510);  mm_208 = _shape_param_510 = None
        convert_element_type_584: "f32[1536, 6144]" = torch.ops.prims.convert_element_type.default(mm_209, torch.float32);  mm_209 = None
        convert_element_type_585: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_583, torch.float32);  convert_element_type_583 = None
        convert_element_type_586: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_506, torch.float32);  view_506 = None
        view_507: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(arg480_1, _shape_param_511);  arg480_1 = _shape_param_511 = None
        convert_element_type_587: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_507, torch.float32);  view_507 = None
        mul_468: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_587, 0.7071067811865476)
        erf_18: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_468);  mul_468 = None
        add_105: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
        mul_469: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(add_105, 0.5);  add_105 = None
        mul_470: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_587, convert_element_type_587)
        mul_471: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_470, -0.5);  mul_470 = None
        exp_36: "f32[8, 512, 6144]" = torch.ops.aten.exp.default(mul_471);  mul_471 = None
        mul_472: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_36, 0.3989422804014327);  exp_36 = None
        mul_473: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_587, mul_472);  convert_element_type_587 = mul_472 = None
        add_106: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(mul_469, mul_473);  mul_469 = mul_473 = None
        mul_474: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_586, add_106);  convert_element_type_586 = add_106 = None
        convert_element_type_588: "bf16[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(mul_474, torch.bfloat16);  mul_474 = None
        view_508: "bf16[4096, 6144]" = torch.ops.aten.view.default(convert_element_type_588, _shape_param_512);  convert_element_type_588 = _shape_param_512 = None
        mm_210: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_508, arg481_1);  arg481_1 = None
        permute_190: "bf16[6144, 4096]" = torch.ops.aten.permute.default(view_508, [1, 0])
        mm_211: "bf16[6144, 1536]" = torch.ops.aten.mm.default(permute_190, arg482_1);  permute_190 = arg482_1 = None
        sum_268: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_508, [0], True, dtype = torch.float32);  view_508 = None
        view_509: "f32[6144]" = torch.ops.aten.view.default(sum_268, _shape_param_513);  sum_268 = _shape_param_513 = None
        convert_element_type_589: "bf16[6144]" = torch.ops.prims.convert_element_type.default(view_509, torch.bfloat16);  view_509 = None
        view_510: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_210, _shape_param_514);  mm_210 = _shape_param_514 = None
        convert_element_type_590: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_510, torch.float32);  view_510 = None
        add_107: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_464, convert_element_type_590);  mul_464 = convert_element_type_590 = None
        convert_element_type_591: "f32[6144, 1536]" = torch.ops.prims.convert_element_type.default(mm_211, torch.float32);  mm_211 = None
        convert_element_type_592: "f32[6144]" = torch.ops.prims.convert_element_type.default(convert_element_type_589, torch.float32);  convert_element_type_589 = None
        mul_475: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_107, arg483_1);  arg483_1 = None
        mul_476: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_475, 1536)
        sum_269: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_475, [2], True)
        mul_477: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_475, arg484_1);  mul_475 = None
        sum_270: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_477, [2], True);  mul_477 = None
        mul_478: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg484_1, sum_270);  sum_270 = None
        sub_93: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_476, sum_269);  mul_476 = sum_269 = None
        sub_94: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_93, mul_478);  sub_93 = mul_478 = None
        mul_479: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg485_1, sub_94);  arg485_1 = sub_94 = None
        mul_480: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_107, arg484_1);  arg484_1 = None
        sum_271: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_480, [0, 1]);  mul_480 = None
        sum_272: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_107, [0, 1]);  add_107 = None
        convert_element_type_593: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(mul_479, torch.bfloat16)
        convert_element_type_594: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(arg486_1, torch.bfloat16);  arg486_1 = None
        mul_481: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_594, 1.1111111111111112);  convert_element_type_594 = None
        mul_482: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_593, mul_481);  convert_element_type_593 = mul_481 = None
        view_511: "bf16[4096, 1536]" = torch.ops.aten.view.default(mul_482, _shape_param_515);  mul_482 = _shape_param_515 = None
        mm_212: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_511, arg487_1);  arg487_1 = None
        permute_191: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_511, [1, 0])
        mm_213: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_191, arg488_1);  permute_191 = arg488_1 = None
        sum_273: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_511, [0], True, dtype = torch.float32);  view_511 = None
        view_512: "f32[1536]" = torch.ops.aten.view.default(sum_273, _shape_param_516);  sum_273 = _shape_param_516 = None
        convert_element_type_595: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_512, torch.bfloat16);  view_512 = None
        view_513: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_212, _shape_param_517);  mm_212 = _shape_param_517 = None
        convert_element_type_596: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_213, torch.float32);  mm_213 = None
        convert_element_type_597: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_595, torch.float32);  convert_element_type_595 = None
        view_514: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_513, _shape_param_518);  view_513 = _shape_param_518 = None
        permute_192: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_514, [0, 2, 1, 3]);  view_514 = None
        clone_68: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_192, memory_format = torch.contiguous_format);  permute_192 = None
        view_515: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_68, _shape_param_519);  clone_68 = _shape_param_519 = None
        bmm_68: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(arg489_1, view_515);  arg489_1 = None
        bmm_69: "bf16[192, 512, 512]" = torch.ops.aten.bmm.default(view_515, arg490_1);  view_515 = arg490_1 = None
        convert_element_type_598: "f32[192, 512, 512]" = torch.ops.prims.convert_element_type.default(bmm_69, torch.float32);  bmm_69 = None
        view_516: "f32[8, 24, 512, 512]" = torch.ops.aten.view.default(convert_element_type_598, _shape_param_520);  convert_element_type_598 = _shape_param_520 = None
        convert_element_type_599: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(arg491_1, torch.float32);  arg491_1 = None
        mul_483: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_599, 1.1111111111111112);  convert_element_type_599 = None
        mul_484: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(view_516, mul_483);  view_516 = mul_483 = None
        convert_element_type_600: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(arg492_1, torch.float32);  arg492_1 = None
        sub_95: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_600, arg493_1);  convert_element_type_600 = arg493_1 = None
        exp_37: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_95);  sub_95 = None
        div_36: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_37, arg494_1);  exp_37 = arg494_1 = None
        mul_485: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_484, div_36);  mul_484 = None
        sum_274: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_485, [-1], True)
        neg_17: "f32[8, 24, 512, 512]" = torch.ops.aten.neg.default(div_36);  div_36 = None
        fma_17: "f32[8, 24, 512, 512]" = torch.ops.prims.fma.default(neg_17, sum_274, mul_485);  neg_17 = sum_274 = mul_485 = None
        convert_element_type_601: "bf16[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(fma_17, torch.bfloat16);  fma_17 = None
        where_20: "bf16[8, 24, 512, 512]" = torch.ops.aten.where.self(full_3, full_2, convert_element_type_601);  convert_element_type_601 = None
        view_517: "bf16[192, 512, 512]" = torch.ops.aten.view.default(where_20, _shape_param_521);  where_20 = _shape_param_521 = None
        bmm_70: "bf16[192, 64, 512]" = torch.ops.aten.bmm.default(arg495_1, view_517);  arg495_1 = None
        bmm_71: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(view_517, arg496_1);  view_517 = arg496_1 = None
        div_37: "bf16[192, 64, 512]" = torch.ops.aten.div.Tensor(bmm_70, full_4);  bmm_70 = None
        permute_193: "bf16[192, 512, 64]" = torch.ops.aten.permute.default(div_37, [0, 2, 1]);  div_37 = None
        view_518: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_68, _shape_param_522);  bmm_68 = _shape_param_522 = None
        permute_194: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_518, [0, 2, 1, 3]);  view_518 = None
        clone_69: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_194, memory_format = torch.contiguous_format);  permute_194 = None
        view_519: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_69, _shape_param_523);  clone_69 = _shape_param_523 = None
        view_520: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_519, _shape_param_524);  view_519 = _shape_param_524 = None
        mm_214: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_520, arg497_1);  arg497_1 = None
        permute_195: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_520, [1, 0])
        mm_215: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_195, arg498_1);  permute_195 = None
        sum_275: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_520, [0], True, dtype = torch.float32);  view_520 = None
        view_521: "f32[1536]" = torch.ops.aten.view.default(sum_275, _shape_param_525);  sum_275 = _shape_param_525 = None
        convert_element_type_602: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_521, torch.bfloat16);  view_521 = None
        view_522: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_214, _shape_param_526);  mm_214 = _shape_param_526 = None
        convert_element_type_603: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_522, torch.float32);  view_522 = None
        add_108: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_479, convert_element_type_603);  mul_479 = convert_element_type_603 = None
        convert_element_type_604: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_215, torch.float32);  mm_215 = None
        convert_element_type_605: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_602, torch.float32);  convert_element_type_602 = None
        view_523: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(permute_193, _shape_param_527);  permute_193 = _shape_param_527 = None
        permute_196: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_523, [0, 2, 1, 3]);  view_523 = None
        view_524: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(permute_196, _shape_param_528);  permute_196 = _shape_param_528 = None
        clone_70: "bf16[8, 512, 1536]" = torch.ops.aten.clone.default(view_524, memory_format = torch.contiguous_format);  view_524 = None
        view_525: "bf16[4096, 1536]" = torch.ops.aten.view.default(clone_70, _shape_param_529);  clone_70 = _shape_param_529 = None
        mm_216: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_525, arg499_1);  arg499_1 = None
        permute_197: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_525, [1, 0])
        mm_217: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_197, arg498_1);  permute_197 = None
        sum_276: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_525, [0], True, dtype = torch.float32);  view_525 = None
        view_526: "f32[1536]" = torch.ops.aten.view.default(sum_276, _shape_param_530);  sum_276 = _shape_param_530 = None
        convert_element_type_606: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_526, torch.bfloat16);  view_526 = None
        view_527: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_216, _shape_param_531);  mm_216 = _shape_param_531 = None
        convert_element_type_607: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_527, torch.float32);  view_527 = None
        add_109: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_108, convert_element_type_607);  add_108 = convert_element_type_607 = None
        convert_element_type_608: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_217, torch.float32);  mm_217 = None
        convert_element_type_609: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_606, torch.float32);  convert_element_type_606 = None
        view_528: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_71, _shape_param_532);  bmm_71 = _shape_param_532 = None
        permute_198: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_528, [0, 2, 1, 3]);  view_528 = None
        clone_71: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_198, memory_format = torch.contiguous_format);  permute_198 = None
        view_529: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_71, _shape_param_533);  clone_71 = _shape_param_533 = None
        view_530: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_529, _shape_param_534);  view_529 = _shape_param_534 = None
        mm_218: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_530, arg500_1);  arg500_1 = None
        permute_199: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_530, [1, 0])
        mm_219: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_199, arg498_1);  permute_199 = arg498_1 = None
        sum_277: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_530, [0], True, dtype = torch.float32);  view_530 = None
        view_531: "f32[1536]" = torch.ops.aten.view.default(sum_277, _shape_param_535);  sum_277 = _shape_param_535 = None
        convert_element_type_610: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_531, torch.bfloat16);  view_531 = None
        view_532: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_218, _shape_param_536);  mm_218 = _shape_param_536 = None
        convert_element_type_611: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_532, torch.float32);  view_532 = None
        add_110: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_109, convert_element_type_611);  add_109 = convert_element_type_611 = None
        convert_element_type_612: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_219, torch.float32);  mm_219 = None
        convert_element_type_613: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_610, torch.float32);  convert_element_type_610 = None
        mul_486: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_110, arg501_1);  arg501_1 = None
        mul_487: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_486, 1536)
        sum_278: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_486, [2], True)
        mul_488: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_486, arg502_1);  mul_486 = None
        sum_279: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_488, [2], True);  mul_488 = None
        mul_489: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg502_1, sum_279);  sum_279 = None
        sub_96: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_487, sum_278);  mul_487 = sum_278 = None
        sub_97: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_96, mul_489);  sub_96 = mul_489 = None
        mul_490: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg503_1, sub_97);  arg503_1 = sub_97 = None
        mul_491: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_110, arg502_1);  arg502_1 = None
        sum_280: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_491, [0, 1]);  mul_491 = None
        sum_281: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_110, [0, 1]);  add_110 = None
        convert_element_type_614: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(mul_490, torch.bfloat16)
        convert_element_type_615: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(arg504_1, torch.bfloat16);  arg504_1 = None
        mul_492: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_615, 1.1111111111111112);  convert_element_type_615 = None
        mul_493: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_614, mul_492);  convert_element_type_614 = mul_492 = None
        view_533: "bf16[4096, 1536]" = torch.ops.aten.view.default(mul_493, _shape_param_537);  mul_493 = _shape_param_537 = None
        mm_220: "bf16[4096, 6144]" = torch.ops.aten.mm.default(view_533, arg505_1);  arg505_1 = None
        permute_200: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_533, [1, 0])
        mm_221: "bf16[1536, 6144]" = torch.ops.aten.mm.default(permute_200, arg506_1);  permute_200 = arg506_1 = None
        sum_282: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_533, [0], True, dtype = torch.float32);  view_533 = None
        view_534: "f32[1536]" = torch.ops.aten.view.default(sum_282, _shape_param_538);  sum_282 = _shape_param_538 = None
        convert_element_type_616: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_534, torch.bfloat16);  view_534 = None
        view_535: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(mm_220, _shape_param_539);  mm_220 = _shape_param_539 = None
        convert_element_type_617: "f32[1536, 6144]" = torch.ops.prims.convert_element_type.default(mm_221, torch.float32);  mm_221 = None
        convert_element_type_618: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_616, torch.float32);  convert_element_type_616 = None
        convert_element_type_619: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_535, torch.float32);  view_535 = None
        view_536: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(arg507_1, _shape_param_540);  arg507_1 = _shape_param_540 = None
        convert_element_type_620: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_536, torch.float32);  view_536 = None
        mul_494: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_620, 0.7071067811865476)
        erf_19: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_494);  mul_494 = None
        add_111: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
        mul_495: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(add_111, 0.5);  add_111 = None
        mul_496: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_620, convert_element_type_620)
        mul_497: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_496, -0.5);  mul_496 = None
        exp_38: "f32[8, 512, 6144]" = torch.ops.aten.exp.default(mul_497);  mul_497 = None
        mul_498: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_38, 0.3989422804014327);  exp_38 = None
        mul_499: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_620, mul_498);  convert_element_type_620 = mul_498 = None
        add_112: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(mul_495, mul_499);  mul_495 = mul_499 = None
        mul_500: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_619, add_112);  convert_element_type_619 = add_112 = None
        convert_element_type_621: "bf16[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(mul_500, torch.bfloat16);  mul_500 = None
        view_537: "bf16[4096, 6144]" = torch.ops.aten.view.default(convert_element_type_621, _shape_param_541);  convert_element_type_621 = _shape_param_541 = None
        mm_222: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_537, arg508_1);  arg508_1 = None
        permute_201: "bf16[6144, 4096]" = torch.ops.aten.permute.default(view_537, [1, 0])
        mm_223: "bf16[6144, 1536]" = torch.ops.aten.mm.default(permute_201, arg509_1);  permute_201 = arg509_1 = None
        sum_283: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_537, [0], True, dtype = torch.float32);  view_537 = None
        view_538: "f32[6144]" = torch.ops.aten.view.default(sum_283, _shape_param_542);  sum_283 = _shape_param_542 = None
        convert_element_type_622: "bf16[6144]" = torch.ops.prims.convert_element_type.default(view_538, torch.bfloat16);  view_538 = None
        view_539: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_222, _shape_param_543);  mm_222 = _shape_param_543 = None
        convert_element_type_623: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_539, torch.float32);  view_539 = None
        add_113: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_490, convert_element_type_623);  mul_490 = convert_element_type_623 = None
        convert_element_type_624: "f32[6144, 1536]" = torch.ops.prims.convert_element_type.default(mm_223, torch.float32);  mm_223 = None
        convert_element_type_625: "f32[6144]" = torch.ops.prims.convert_element_type.default(convert_element_type_622, torch.float32);  convert_element_type_622 = None
        mul_501: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_113, arg510_1);  arg510_1 = None
        mul_502: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_501, 1536)
        sum_284: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_501, [2], True)
        mul_503: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_501, arg511_1);  mul_501 = None
        sum_285: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_503, [2], True);  mul_503 = None
        mul_504: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg511_1, sum_285);  sum_285 = None
        sub_98: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_502, sum_284);  mul_502 = sum_284 = None
        sub_99: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_98, mul_504);  sub_98 = mul_504 = None
        mul_505: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg512_1, sub_99);  arg512_1 = sub_99 = None
        mul_506: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_113, arg511_1);  arg511_1 = None
        sum_286: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_506, [0, 1]);  mul_506 = None
        sum_287: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_113, [0, 1]);  add_113 = None
        convert_element_type_626: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(mul_505, torch.bfloat16)
        convert_element_type_627: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(arg513_1, torch.bfloat16);  arg513_1 = None
        mul_507: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_627, 1.1111111111111112);  convert_element_type_627 = None
        mul_508: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_626, mul_507);  convert_element_type_626 = mul_507 = None
        view_540: "bf16[4096, 1536]" = torch.ops.aten.view.default(mul_508, _shape_param_544);  mul_508 = _shape_param_544 = None
        mm_224: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_540, arg514_1);  arg514_1 = None
        permute_202: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_540, [1, 0])
        mm_225: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_202, arg515_1);  permute_202 = arg515_1 = None
        sum_288: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_540, [0], True, dtype = torch.float32);  view_540 = None
        view_541: "f32[1536]" = torch.ops.aten.view.default(sum_288, _shape_param_545);  sum_288 = _shape_param_545 = None
        convert_element_type_628: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_541, torch.bfloat16);  view_541 = None
        view_542: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_224, _shape_param_546);  mm_224 = _shape_param_546 = None
        convert_element_type_629: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_225, torch.float32);  mm_225 = None
        convert_element_type_630: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_628, torch.float32);  convert_element_type_628 = None
        view_543: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_542, _shape_param_547);  view_542 = _shape_param_547 = None
        permute_203: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_543, [0, 2, 1, 3]);  view_543 = None
        clone_72: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_203, memory_format = torch.contiguous_format);  permute_203 = None
        view_544: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_72, _shape_param_548);  clone_72 = _shape_param_548 = None
        bmm_72: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(arg516_1, view_544);  arg516_1 = None
        bmm_73: "bf16[192, 512, 512]" = torch.ops.aten.bmm.default(view_544, arg517_1);  view_544 = arg517_1 = None
        convert_element_type_631: "f32[192, 512, 512]" = torch.ops.prims.convert_element_type.default(bmm_73, torch.float32);  bmm_73 = None
        view_545: "f32[8, 24, 512, 512]" = torch.ops.aten.view.default(convert_element_type_631, _shape_param_549);  convert_element_type_631 = _shape_param_549 = None
        convert_element_type_632: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(arg518_1, torch.float32);  arg518_1 = None
        mul_509: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_632, 1.1111111111111112);  convert_element_type_632 = None
        mul_510: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(view_545, mul_509);  view_545 = mul_509 = None
        convert_element_type_633: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(arg519_1, torch.float32);  arg519_1 = None
        sub_100: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_633, arg520_1);  convert_element_type_633 = arg520_1 = None
        exp_39: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_100);  sub_100 = None
        div_38: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_39, arg521_1);  exp_39 = arg521_1 = None
        mul_511: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_510, div_38);  mul_510 = None
        sum_289: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_511, [-1], True)
        neg_18: "f32[8, 24, 512, 512]" = torch.ops.aten.neg.default(div_38);  div_38 = None
        fma_18: "f32[8, 24, 512, 512]" = torch.ops.prims.fma.default(neg_18, sum_289, mul_511);  neg_18 = sum_289 = mul_511 = None
        convert_element_type_634: "bf16[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(fma_18, torch.bfloat16);  fma_18 = None
        where_21: "bf16[8, 24, 512, 512]" = torch.ops.aten.where.self(full_3, full_2, convert_element_type_634);  convert_element_type_634 = None
        view_546: "bf16[192, 512, 512]" = torch.ops.aten.view.default(where_21, _shape_param_550);  where_21 = _shape_param_550 = None
        bmm_74: "bf16[192, 64, 512]" = torch.ops.aten.bmm.default(arg522_1, view_546);  arg522_1 = None
        bmm_75: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(view_546, arg523_1);  view_546 = arg523_1 = None
        div_39: "bf16[192, 64, 512]" = torch.ops.aten.div.Tensor(bmm_74, full_4);  bmm_74 = None
        permute_204: "bf16[192, 512, 64]" = torch.ops.aten.permute.default(div_39, [0, 2, 1]);  div_39 = None
        view_547: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_72, _shape_param_551);  bmm_72 = _shape_param_551 = None
        permute_205: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_547, [0, 2, 1, 3]);  view_547 = None
        clone_73: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_205, memory_format = torch.contiguous_format);  permute_205 = None
        view_548: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_73, _shape_param_552);  clone_73 = _shape_param_552 = None
        view_549: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_548, _shape_param_553);  view_548 = _shape_param_553 = None
        mm_226: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_549, arg524_1);  arg524_1 = None
        permute_206: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_549, [1, 0])
        mm_227: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_206, arg525_1);  permute_206 = None
        sum_290: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_549, [0], True, dtype = torch.float32);  view_549 = None
        view_550: "f32[1536]" = torch.ops.aten.view.default(sum_290, _shape_param_554);  sum_290 = _shape_param_554 = None
        convert_element_type_635: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_550, torch.bfloat16);  view_550 = None
        view_551: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_226, _shape_param_555);  mm_226 = _shape_param_555 = None
        convert_element_type_636: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_551, torch.float32);  view_551 = None
        add_114: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_505, convert_element_type_636);  mul_505 = convert_element_type_636 = None
        convert_element_type_637: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_227, torch.float32);  mm_227 = None
        convert_element_type_638: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_635, torch.float32);  convert_element_type_635 = None
        view_552: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(permute_204, _shape_param_556);  permute_204 = _shape_param_556 = None
        permute_207: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_552, [0, 2, 1, 3]);  view_552 = None
        view_553: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(permute_207, _shape_param_557);  permute_207 = _shape_param_557 = None
        clone_74: "bf16[8, 512, 1536]" = torch.ops.aten.clone.default(view_553, memory_format = torch.contiguous_format);  view_553 = None
        view_554: "bf16[4096, 1536]" = torch.ops.aten.view.default(clone_74, _shape_param_558);  clone_74 = _shape_param_558 = None
        mm_228: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_554, arg526_1);  arg526_1 = None
        permute_208: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_554, [1, 0])
        mm_229: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_208, arg525_1);  permute_208 = None
        sum_291: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_554, [0], True, dtype = torch.float32);  view_554 = None
        view_555: "f32[1536]" = torch.ops.aten.view.default(sum_291, _shape_param_559);  sum_291 = _shape_param_559 = None
        convert_element_type_639: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_555, torch.bfloat16);  view_555 = None
        view_556: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_228, _shape_param_560);  mm_228 = _shape_param_560 = None
        convert_element_type_640: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_556, torch.float32);  view_556 = None
        add_115: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_114, convert_element_type_640);  add_114 = convert_element_type_640 = None
        convert_element_type_641: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_229, torch.float32);  mm_229 = None
        convert_element_type_642: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_639, torch.float32);  convert_element_type_639 = None
        view_557: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_75, _shape_param_561);  bmm_75 = _shape_param_561 = None
        permute_209: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_557, [0, 2, 1, 3]);  view_557 = None
        clone_75: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_209, memory_format = torch.contiguous_format);  permute_209 = None
        view_558: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_75, _shape_param_562);  clone_75 = _shape_param_562 = None
        view_559: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_558, _shape_param_563);  view_558 = _shape_param_563 = None
        mm_230: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_559, arg527_1);  arg527_1 = None
        permute_210: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_559, [1, 0])
        mm_231: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_210, arg525_1);  permute_210 = arg525_1 = None
        sum_292: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_559, [0], True, dtype = torch.float32);  view_559 = None
        view_560: "f32[1536]" = torch.ops.aten.view.default(sum_292, _shape_param_564);  sum_292 = _shape_param_564 = None
        convert_element_type_643: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_560, torch.bfloat16);  view_560 = None
        view_561: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_230, _shape_param_565);  mm_230 = _shape_param_565 = None
        convert_element_type_644: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_561, torch.float32);  view_561 = None
        add_116: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_115, convert_element_type_644);  add_115 = convert_element_type_644 = None
        convert_element_type_645: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_231, torch.float32);  mm_231 = None
        convert_element_type_646: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_643, torch.float32);  convert_element_type_643 = None
        mul_512: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_116, arg528_1);  arg528_1 = None
        mul_513: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_512, 1536)
        sum_293: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_512, [2], True)
        mul_514: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_512, arg529_1);  mul_512 = None
        sum_294: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_514, [2], True);  mul_514 = None
        mul_515: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg529_1, sum_294);  sum_294 = None
        sub_101: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_513, sum_293);  mul_513 = sum_293 = None
        sub_102: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_101, mul_515);  sub_101 = mul_515 = None
        mul_516: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg530_1, sub_102);  arg530_1 = sub_102 = None
        mul_517: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_116, arg529_1);  arg529_1 = None
        sum_295: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_517, [0, 1]);  mul_517 = None
        sum_296: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_116, [0, 1]);  add_116 = None
        convert_element_type_647: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(mul_516, torch.bfloat16)
        convert_element_type_648: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(arg531_1, torch.bfloat16);  arg531_1 = None
        mul_518: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_648, 1.1111111111111112);  convert_element_type_648 = None
        mul_519: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_647, mul_518);  convert_element_type_647 = mul_518 = None
        view_562: "bf16[4096, 1536]" = torch.ops.aten.view.default(mul_519, _shape_param_566);  mul_519 = _shape_param_566 = None
        mm_232: "bf16[4096, 6144]" = torch.ops.aten.mm.default(view_562, arg532_1);  arg532_1 = None
        permute_211: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_562, [1, 0])
        mm_233: "bf16[1536, 6144]" = torch.ops.aten.mm.default(permute_211, arg533_1);  permute_211 = arg533_1 = None
        sum_297: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_562, [0], True, dtype = torch.float32);  view_562 = None
        view_563: "f32[1536]" = torch.ops.aten.view.default(sum_297, _shape_param_567);  sum_297 = _shape_param_567 = None
        convert_element_type_649: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_563, torch.bfloat16);  view_563 = None
        view_564: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(mm_232, _shape_param_568);  mm_232 = _shape_param_568 = None
        convert_element_type_650: "f32[1536, 6144]" = torch.ops.prims.convert_element_type.default(mm_233, torch.float32);  mm_233 = None
        convert_element_type_651: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_649, torch.float32);  convert_element_type_649 = None
        convert_element_type_652: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_564, torch.float32);  view_564 = None
        view_565: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(arg534_1, _shape_param_569);  arg534_1 = _shape_param_569 = None
        convert_element_type_653: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_565, torch.float32);  view_565 = None
        mul_520: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_653, 0.7071067811865476)
        erf_20: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_520);  mul_520 = None
        add_117: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
        mul_521: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(add_117, 0.5);  add_117 = None
        mul_522: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_653, convert_element_type_653)
        mul_523: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_522, -0.5);  mul_522 = None
        exp_40: "f32[8, 512, 6144]" = torch.ops.aten.exp.default(mul_523);  mul_523 = None
        mul_524: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_40, 0.3989422804014327);  exp_40 = None
        mul_525: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_653, mul_524);  convert_element_type_653 = mul_524 = None
        add_118: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(mul_521, mul_525);  mul_521 = mul_525 = None
        mul_526: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_652, add_118);  convert_element_type_652 = add_118 = None
        convert_element_type_654: "bf16[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(mul_526, torch.bfloat16);  mul_526 = None
        view_566: "bf16[4096, 6144]" = torch.ops.aten.view.default(convert_element_type_654, _shape_param_570);  convert_element_type_654 = _shape_param_570 = None
        mm_234: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_566, arg535_1);  arg535_1 = None
        permute_212: "bf16[6144, 4096]" = torch.ops.aten.permute.default(view_566, [1, 0])
        mm_235: "bf16[6144, 1536]" = torch.ops.aten.mm.default(permute_212, arg536_1);  permute_212 = arg536_1 = None
        sum_298: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_566, [0], True, dtype = torch.float32);  view_566 = None
        view_567: "f32[6144]" = torch.ops.aten.view.default(sum_298, _shape_param_571);  sum_298 = _shape_param_571 = None
        convert_element_type_655: "bf16[6144]" = torch.ops.prims.convert_element_type.default(view_567, torch.bfloat16);  view_567 = None
        view_568: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_234, _shape_param_572);  mm_234 = _shape_param_572 = None
        convert_element_type_656: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_568, torch.float32);  view_568 = None
        add_119: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_516, convert_element_type_656);  mul_516 = convert_element_type_656 = None
        convert_element_type_657: "f32[6144, 1536]" = torch.ops.prims.convert_element_type.default(mm_235, torch.float32);  mm_235 = None
        convert_element_type_658: "f32[6144]" = torch.ops.prims.convert_element_type.default(convert_element_type_655, torch.float32);  convert_element_type_655 = None
        mul_527: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_119, arg537_1);  arg537_1 = None
        mul_528: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_527, 1536)
        sum_299: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_527, [2], True)
        mul_529: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_527, arg538_1);  mul_527 = None
        sum_300: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_529, [2], True);  mul_529 = None
        mul_530: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg538_1, sum_300);  sum_300 = None
        sub_103: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_528, sum_299);  mul_528 = sum_299 = None
        sub_104: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_103, mul_530);  sub_103 = mul_530 = None
        mul_531: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg539_1, sub_104);  arg539_1 = sub_104 = None
        mul_532: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_119, arg538_1);  arg538_1 = None
        sum_301: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_532, [0, 1]);  mul_532 = None
        sum_302: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_119, [0, 1]);  add_119 = None
        convert_element_type_659: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(mul_531, torch.bfloat16)
        convert_element_type_660: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(arg540_1, torch.bfloat16);  arg540_1 = None
        mul_533: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_660, 1.1111111111111112);  convert_element_type_660 = None
        mul_534: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_659, mul_533);  convert_element_type_659 = mul_533 = None
        view_569: "bf16[4096, 1536]" = torch.ops.aten.view.default(mul_534, _shape_param_573);  mul_534 = _shape_param_573 = None
        mm_236: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_569, arg541_1);  arg541_1 = None
        permute_213: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_569, [1, 0])
        mm_237: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_213, arg542_1);  permute_213 = arg542_1 = None
        sum_303: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_569, [0], True, dtype = torch.float32);  view_569 = None
        view_570: "f32[1536]" = torch.ops.aten.view.default(sum_303, _shape_param_574);  sum_303 = _shape_param_574 = None
        convert_element_type_661: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_570, torch.bfloat16);  view_570 = None
        view_571: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_236, _shape_param_575);  mm_236 = _shape_param_575 = None
        convert_element_type_662: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_237, torch.float32);  mm_237 = None
        convert_element_type_663: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_661, torch.float32);  convert_element_type_661 = None
        view_572: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_571, _shape_param_576);  view_571 = _shape_param_576 = None
        permute_214: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_572, [0, 2, 1, 3]);  view_572 = None
        clone_76: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_214, memory_format = torch.contiguous_format);  permute_214 = None
        view_573: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_76, _shape_param_577);  clone_76 = _shape_param_577 = None
        bmm_76: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(arg543_1, view_573);  arg543_1 = None
        bmm_77: "bf16[192, 512, 512]" = torch.ops.aten.bmm.default(view_573, arg544_1);  view_573 = arg544_1 = None
        convert_element_type_664: "f32[192, 512, 512]" = torch.ops.prims.convert_element_type.default(bmm_77, torch.float32);  bmm_77 = None
        view_574: "f32[8, 24, 512, 512]" = torch.ops.aten.view.default(convert_element_type_664, _shape_param_578);  convert_element_type_664 = _shape_param_578 = None
        convert_element_type_665: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(arg545_1, torch.float32);  arg545_1 = None
        mul_535: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_665, 1.1111111111111112);  convert_element_type_665 = None
        mul_536: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(view_574, mul_535);  view_574 = mul_535 = None
        convert_element_type_666: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(arg546_1, torch.float32);  arg546_1 = None
        sub_105: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_666, arg547_1);  convert_element_type_666 = arg547_1 = None
        exp_41: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_105);  sub_105 = None
        div_40: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_41, arg548_1);  exp_41 = arg548_1 = None
        mul_537: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_536, div_40);  mul_536 = None
        sum_304: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_537, [-1], True)
        neg_19: "f32[8, 24, 512, 512]" = torch.ops.aten.neg.default(div_40);  div_40 = None
        fma_19: "f32[8, 24, 512, 512]" = torch.ops.prims.fma.default(neg_19, sum_304, mul_537);  neg_19 = sum_304 = mul_537 = None
        convert_element_type_667: "bf16[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(fma_19, torch.bfloat16);  fma_19 = None
        where_22: "bf16[8, 24, 512, 512]" = torch.ops.aten.where.self(full_3, full_2, convert_element_type_667);  convert_element_type_667 = None
        view_575: "bf16[192, 512, 512]" = torch.ops.aten.view.default(where_22, _shape_param_579);  where_22 = _shape_param_579 = None
        bmm_78: "bf16[192, 64, 512]" = torch.ops.aten.bmm.default(arg549_1, view_575);  arg549_1 = None
        bmm_79: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(view_575, arg550_1);  view_575 = arg550_1 = None
        div_41: "bf16[192, 64, 512]" = torch.ops.aten.div.Tensor(bmm_78, full_4);  bmm_78 = None
        permute_215: "bf16[192, 512, 64]" = torch.ops.aten.permute.default(div_41, [0, 2, 1]);  div_41 = None
        view_576: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_76, _shape_param_580);  bmm_76 = _shape_param_580 = None
        permute_216: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_576, [0, 2, 1, 3]);  view_576 = None
        clone_77: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_216, memory_format = torch.contiguous_format);  permute_216 = None
        view_577: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_77, _shape_param_581);  clone_77 = _shape_param_581 = None
        view_578: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_577, _shape_param_582);  view_577 = _shape_param_582 = None
        mm_238: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_578, arg551_1);  arg551_1 = None
        permute_217: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_578, [1, 0])
        mm_239: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_217, arg552_1);  permute_217 = None
        sum_305: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_578, [0], True, dtype = torch.float32);  view_578 = None
        view_579: "f32[1536]" = torch.ops.aten.view.default(sum_305, _shape_param_583);  sum_305 = _shape_param_583 = None
        convert_element_type_668: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_579, torch.bfloat16);  view_579 = None
        view_580: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_238, _shape_param_584);  mm_238 = _shape_param_584 = None
        convert_element_type_669: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_580, torch.float32);  view_580 = None
        add_120: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_531, convert_element_type_669);  mul_531 = convert_element_type_669 = None
        convert_element_type_670: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_239, torch.float32);  mm_239 = None
        convert_element_type_671: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_668, torch.float32);  convert_element_type_668 = None
        view_581: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(permute_215, _shape_param_585);  permute_215 = _shape_param_585 = None
        permute_218: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_581, [0, 2, 1, 3]);  view_581 = None
        view_582: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(permute_218, _shape_param_586);  permute_218 = _shape_param_586 = None
        clone_78: "bf16[8, 512, 1536]" = torch.ops.aten.clone.default(view_582, memory_format = torch.contiguous_format);  view_582 = None
        view_583: "bf16[4096, 1536]" = torch.ops.aten.view.default(clone_78, _shape_param_587);  clone_78 = _shape_param_587 = None
        mm_240: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_583, arg553_1);  arg553_1 = None
        permute_219: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_583, [1, 0])
        mm_241: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_219, arg552_1);  permute_219 = None
        sum_306: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_583, [0], True, dtype = torch.float32);  view_583 = None
        view_584: "f32[1536]" = torch.ops.aten.view.default(sum_306, _shape_param_588);  sum_306 = _shape_param_588 = None
        convert_element_type_672: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_584, torch.bfloat16);  view_584 = None
        view_585: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_240, _shape_param_589);  mm_240 = _shape_param_589 = None
        convert_element_type_673: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_585, torch.float32);  view_585 = None
        add_121: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_120, convert_element_type_673);  add_120 = convert_element_type_673 = None
        convert_element_type_674: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_241, torch.float32);  mm_241 = None
        convert_element_type_675: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_672, torch.float32);  convert_element_type_672 = None
        view_586: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_79, _shape_param_590);  bmm_79 = _shape_param_590 = None
        permute_220: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_586, [0, 2, 1, 3]);  view_586 = None
        clone_79: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_220, memory_format = torch.contiguous_format);  permute_220 = None
        view_587: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_79, _shape_param_591);  clone_79 = _shape_param_591 = None
        view_588: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_587, _shape_param_592);  view_587 = _shape_param_592 = None
        mm_242: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_588, arg554_1);  arg554_1 = None
        permute_221: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_588, [1, 0])
        mm_243: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_221, arg552_1);  permute_221 = arg552_1 = None
        sum_307: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_588, [0], True, dtype = torch.float32);  view_588 = None
        view_589: "f32[1536]" = torch.ops.aten.view.default(sum_307, _shape_param_593);  sum_307 = _shape_param_593 = None
        convert_element_type_676: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_589, torch.bfloat16);  view_589 = None
        view_590: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_242, _shape_param_594);  mm_242 = _shape_param_594 = None
        convert_element_type_677: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_590, torch.float32);  view_590 = None
        add_122: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_121, convert_element_type_677);  add_121 = convert_element_type_677 = None
        convert_element_type_678: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_243, torch.float32);  mm_243 = None
        convert_element_type_679: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_676, torch.float32);  convert_element_type_676 = None
        mul_538: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_122, arg555_1);  arg555_1 = None
        mul_539: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_538, 1536)
        sum_308: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_538, [2], True)
        mul_540: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_538, arg556_1);  mul_538 = None
        sum_309: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_540, [2], True);  mul_540 = None
        mul_541: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg556_1, sum_309);  sum_309 = None
        sub_106: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_539, sum_308);  mul_539 = sum_308 = None
        sub_107: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_106, mul_541);  sub_106 = mul_541 = None
        mul_542: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg557_1, sub_107);  arg557_1 = sub_107 = None
        mul_543: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_122, arg556_1);  arg556_1 = None
        sum_310: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_543, [0, 1]);  mul_543 = None
        sum_311: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_122, [0, 1]);  add_122 = None
        convert_element_type_680: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(mul_542, torch.bfloat16)
        convert_element_type_681: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(arg558_1, torch.bfloat16);  arg558_1 = None
        mul_544: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_681, 1.1111111111111112);  convert_element_type_681 = None
        mul_545: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_680, mul_544);  convert_element_type_680 = mul_544 = None
        view_591: "bf16[4096, 1536]" = torch.ops.aten.view.default(mul_545, _shape_param_595);  mul_545 = _shape_param_595 = None
        mm_244: "bf16[4096, 6144]" = torch.ops.aten.mm.default(view_591, arg559_1);  arg559_1 = None
        permute_222: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_591, [1, 0])
        mm_245: "bf16[1536, 6144]" = torch.ops.aten.mm.default(permute_222, arg560_1);  permute_222 = arg560_1 = None
        sum_312: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_591, [0], True, dtype = torch.float32);  view_591 = None
        view_592: "f32[1536]" = torch.ops.aten.view.default(sum_312, _shape_param_596);  sum_312 = _shape_param_596 = None
        convert_element_type_682: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_592, torch.bfloat16);  view_592 = None
        view_593: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(mm_244, _shape_param_597);  mm_244 = _shape_param_597 = None
        convert_element_type_683: "f32[1536, 6144]" = torch.ops.prims.convert_element_type.default(mm_245, torch.float32);  mm_245 = None
        convert_element_type_684: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_682, torch.float32);  convert_element_type_682 = None
        convert_element_type_685: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_593, torch.float32);  view_593 = None
        view_594: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(arg561_1, _shape_param_598);  arg561_1 = _shape_param_598 = None
        convert_element_type_686: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_594, torch.float32);  view_594 = None
        mul_546: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_686, 0.7071067811865476)
        erf_21: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_546);  mul_546 = None
        add_123: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
        mul_547: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(add_123, 0.5);  add_123 = None
        mul_548: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_686, convert_element_type_686)
        mul_549: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_548, -0.5);  mul_548 = None
        exp_42: "f32[8, 512, 6144]" = torch.ops.aten.exp.default(mul_549);  mul_549 = None
        mul_550: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_42, 0.3989422804014327);  exp_42 = None
        mul_551: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_686, mul_550);  convert_element_type_686 = mul_550 = None
        add_124: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(mul_547, mul_551);  mul_547 = mul_551 = None
        mul_552: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_685, add_124);  convert_element_type_685 = add_124 = None
        convert_element_type_687: "bf16[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(mul_552, torch.bfloat16);  mul_552 = None
        view_595: "bf16[4096, 6144]" = torch.ops.aten.view.default(convert_element_type_687, _shape_param_599);  convert_element_type_687 = _shape_param_599 = None
        mm_246: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_595, arg562_1);  arg562_1 = None
        permute_223: "bf16[6144, 4096]" = torch.ops.aten.permute.default(view_595, [1, 0])
        mm_247: "bf16[6144, 1536]" = torch.ops.aten.mm.default(permute_223, arg563_1);  permute_223 = arg563_1 = None
        sum_313: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_595, [0], True, dtype = torch.float32);  view_595 = None
        view_596: "f32[6144]" = torch.ops.aten.view.default(sum_313, _shape_param_600);  sum_313 = _shape_param_600 = None
        convert_element_type_688: "bf16[6144]" = torch.ops.prims.convert_element_type.default(view_596, torch.bfloat16);  view_596 = None
        view_597: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_246, _shape_param_601);  mm_246 = _shape_param_601 = None
        convert_element_type_689: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_597, torch.float32);  view_597 = None
        add_125: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_542, convert_element_type_689);  mul_542 = convert_element_type_689 = None
        convert_element_type_690: "f32[6144, 1536]" = torch.ops.prims.convert_element_type.default(mm_247, torch.float32);  mm_247 = None
        convert_element_type_691: "f32[6144]" = torch.ops.prims.convert_element_type.default(convert_element_type_688, torch.float32);  convert_element_type_688 = None
        mul_553: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_125, arg564_1);  arg564_1 = None
        mul_554: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_553, 1536)
        sum_314: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_553, [2], True)
        mul_555: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_553, arg565_1);  mul_553 = None
        sum_315: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_555, [2], True);  mul_555 = None
        mul_556: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg565_1, sum_315);  sum_315 = None
        sub_108: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_554, sum_314);  mul_554 = sum_314 = None
        sub_109: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_108, mul_556);  sub_108 = mul_556 = None
        mul_557: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg566_1, sub_109);  arg566_1 = sub_109 = None
        mul_558: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_125, arg565_1);  arg565_1 = None
        sum_316: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_558, [0, 1]);  mul_558 = None
        sum_317: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_125, [0, 1]);  add_125 = None
        convert_element_type_692: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(mul_557, torch.bfloat16)
        convert_element_type_693: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(arg567_1, torch.bfloat16);  arg567_1 = None
        mul_559: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_693, 1.1111111111111112);  convert_element_type_693 = None
        mul_560: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_692, mul_559);  convert_element_type_692 = mul_559 = None
        view_598: "bf16[4096, 1536]" = torch.ops.aten.view.default(mul_560, _shape_param_602);  mul_560 = _shape_param_602 = None
        mm_248: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_598, arg568_1);  arg568_1 = None
        permute_224: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_598, [1, 0])
        mm_249: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_224, arg569_1);  permute_224 = arg569_1 = None
        sum_318: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_598, [0], True, dtype = torch.float32);  view_598 = None
        view_599: "f32[1536]" = torch.ops.aten.view.default(sum_318, _shape_param_603);  sum_318 = _shape_param_603 = None
        convert_element_type_694: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_599, torch.bfloat16);  view_599 = None
        view_600: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_248, _shape_param_604);  mm_248 = _shape_param_604 = None
        convert_element_type_695: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_249, torch.float32);  mm_249 = None
        convert_element_type_696: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_694, torch.float32);  convert_element_type_694 = None
        view_601: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_600, _shape_param_605);  view_600 = _shape_param_605 = None
        permute_225: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_601, [0, 2, 1, 3]);  view_601 = None
        clone_80: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_225, memory_format = torch.contiguous_format);  permute_225 = None
        view_602: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_80, _shape_param_606);  clone_80 = _shape_param_606 = None
        bmm_80: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(arg570_1, view_602);  arg570_1 = None
        bmm_81: "bf16[192, 512, 512]" = torch.ops.aten.bmm.default(view_602, arg571_1);  view_602 = arg571_1 = None
        convert_element_type_697: "f32[192, 512, 512]" = torch.ops.prims.convert_element_type.default(bmm_81, torch.float32);  bmm_81 = None
        view_603: "f32[8, 24, 512, 512]" = torch.ops.aten.view.default(convert_element_type_697, _shape_param_607);  convert_element_type_697 = _shape_param_607 = None
        convert_element_type_698: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(arg572_1, torch.float32);  arg572_1 = None
        mul_561: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_698, 1.1111111111111112);  convert_element_type_698 = None
        mul_562: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(view_603, mul_561);  view_603 = mul_561 = None
        convert_element_type_699: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(arg573_1, torch.float32);  arg573_1 = None
        sub_110: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_699, arg574_1);  convert_element_type_699 = arg574_1 = None
        exp_43: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_110);  sub_110 = None
        div_42: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_43, arg575_1);  exp_43 = arg575_1 = None
        mul_563: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_562, div_42);  mul_562 = None
        sum_319: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_563, [-1], True)
        neg_20: "f32[8, 24, 512, 512]" = torch.ops.aten.neg.default(div_42);  div_42 = None
        fma_20: "f32[8, 24, 512, 512]" = torch.ops.prims.fma.default(neg_20, sum_319, mul_563);  neg_20 = sum_319 = mul_563 = None
        convert_element_type_700: "bf16[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(fma_20, torch.bfloat16);  fma_20 = None
        where_23: "bf16[8, 24, 512, 512]" = torch.ops.aten.where.self(full_3, full_2, convert_element_type_700);  convert_element_type_700 = None
        view_604: "bf16[192, 512, 512]" = torch.ops.aten.view.default(where_23, _shape_param_608);  where_23 = _shape_param_608 = None
        bmm_82: "bf16[192, 64, 512]" = torch.ops.aten.bmm.default(arg576_1, view_604);  arg576_1 = None
        bmm_83: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(view_604, arg577_1);  view_604 = arg577_1 = None
        div_43: "bf16[192, 64, 512]" = torch.ops.aten.div.Tensor(bmm_82, full_4);  bmm_82 = None
        permute_226: "bf16[192, 512, 64]" = torch.ops.aten.permute.default(div_43, [0, 2, 1]);  div_43 = None
        view_605: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_80, _shape_param_609);  bmm_80 = _shape_param_609 = None
        permute_227: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_605, [0, 2, 1, 3]);  view_605 = None
        clone_81: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_227, memory_format = torch.contiguous_format);  permute_227 = None
        view_606: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_81, _shape_param_610);  clone_81 = _shape_param_610 = None
        view_607: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_606, _shape_param_611);  view_606 = _shape_param_611 = None
        mm_250: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_607, arg578_1);  arg578_1 = None
        permute_228: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_607, [1, 0])
        mm_251: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_228, arg579_1);  permute_228 = None
        sum_320: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_607, [0], True, dtype = torch.float32);  view_607 = None
        view_608: "f32[1536]" = torch.ops.aten.view.default(sum_320, _shape_param_612);  sum_320 = _shape_param_612 = None
        convert_element_type_701: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_608, torch.bfloat16);  view_608 = None
        view_609: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_250, _shape_param_613);  mm_250 = _shape_param_613 = None
        convert_element_type_702: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_609, torch.float32);  view_609 = None
        add_126: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_557, convert_element_type_702);  mul_557 = convert_element_type_702 = None
        convert_element_type_703: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_251, torch.float32);  mm_251 = None
        convert_element_type_704: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_701, torch.float32);  convert_element_type_701 = None
        view_610: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(permute_226, _shape_param_614);  permute_226 = _shape_param_614 = None
        permute_229: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_610, [0, 2, 1, 3]);  view_610 = None
        view_611: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(permute_229, _shape_param_615);  permute_229 = _shape_param_615 = None
        clone_82: "bf16[8, 512, 1536]" = torch.ops.aten.clone.default(view_611, memory_format = torch.contiguous_format);  view_611 = None
        view_612: "bf16[4096, 1536]" = torch.ops.aten.view.default(clone_82, _shape_param_616);  clone_82 = _shape_param_616 = None
        mm_252: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_612, arg580_1);  arg580_1 = None
        permute_230: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_612, [1, 0])
        mm_253: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_230, arg579_1);  permute_230 = None
        sum_321: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_612, [0], True, dtype = torch.float32);  view_612 = None
        view_613: "f32[1536]" = torch.ops.aten.view.default(sum_321, _shape_param_617);  sum_321 = _shape_param_617 = None
        convert_element_type_705: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_613, torch.bfloat16);  view_613 = None
        view_614: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_252, _shape_param_618);  mm_252 = _shape_param_618 = None
        convert_element_type_706: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_614, torch.float32);  view_614 = None
        add_127: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_126, convert_element_type_706);  add_126 = convert_element_type_706 = None
        convert_element_type_707: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_253, torch.float32);  mm_253 = None
        convert_element_type_708: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_705, torch.float32);  convert_element_type_705 = None
        view_615: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_83, _shape_param_619);  bmm_83 = _shape_param_619 = None
        permute_231: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_615, [0, 2, 1, 3]);  view_615 = None
        clone_83: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_231, memory_format = torch.contiguous_format);  permute_231 = None
        view_616: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_83, _shape_param_620);  clone_83 = _shape_param_620 = None
        view_617: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_616, _shape_param_621);  view_616 = _shape_param_621 = None
        mm_254: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_617, arg581_1);  arg581_1 = None
        permute_232: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_617, [1, 0])
        mm_255: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_232, arg579_1);  permute_232 = arg579_1 = None
        sum_322: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_617, [0], True, dtype = torch.float32);  view_617 = None
        view_618: "f32[1536]" = torch.ops.aten.view.default(sum_322, _shape_param_622);  sum_322 = _shape_param_622 = None
        convert_element_type_709: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_618, torch.bfloat16);  view_618 = None
        view_619: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_254, _shape_param_623);  mm_254 = _shape_param_623 = None
        convert_element_type_710: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_619, torch.float32);  view_619 = None
        add_128: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_127, convert_element_type_710);  add_127 = convert_element_type_710 = None
        convert_element_type_711: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_255, torch.float32);  mm_255 = None
        convert_element_type_712: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_709, torch.float32);  convert_element_type_709 = None
        mul_564: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_128, arg582_1);  arg582_1 = None
        mul_565: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_564, 1536)
        sum_323: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_564, [2], True)
        mul_566: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_564, arg583_1);  mul_564 = None
        sum_324: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_566, [2], True);  mul_566 = None
        mul_567: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg583_1, sum_324);  sum_324 = None
        sub_111: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_565, sum_323);  mul_565 = sum_323 = None
        sub_112: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_111, mul_567);  sub_111 = mul_567 = None
        mul_568: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg584_1, sub_112);  arg584_1 = sub_112 = None
        mul_569: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_128, arg583_1);  arg583_1 = None
        sum_325: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_569, [0, 1]);  mul_569 = None
        sum_326: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_128, [0, 1]);  add_128 = None
        convert_element_type_713: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(mul_568, torch.bfloat16)
        convert_element_type_714: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(arg585_1, torch.bfloat16);  arg585_1 = None
        mul_570: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_714, 1.1111111111111112);  convert_element_type_714 = None
        mul_571: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_713, mul_570);  convert_element_type_713 = mul_570 = None
        view_620: "bf16[4096, 1536]" = torch.ops.aten.view.default(mul_571, _shape_param_624);  mul_571 = _shape_param_624 = None
        mm_256: "bf16[4096, 6144]" = torch.ops.aten.mm.default(view_620, arg586_1);  arg586_1 = None
        permute_233: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_620, [1, 0])
        mm_257: "bf16[1536, 6144]" = torch.ops.aten.mm.default(permute_233, arg587_1);  permute_233 = arg587_1 = None
        sum_327: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_620, [0], True, dtype = torch.float32);  view_620 = None
        view_621: "f32[1536]" = torch.ops.aten.view.default(sum_327, _shape_param_625);  sum_327 = _shape_param_625 = None
        convert_element_type_715: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_621, torch.bfloat16);  view_621 = None
        view_622: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(mm_256, _shape_param_626);  mm_256 = _shape_param_626 = None
        convert_element_type_716: "f32[1536, 6144]" = torch.ops.prims.convert_element_type.default(mm_257, torch.float32);  mm_257 = None
        convert_element_type_717: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_715, torch.float32);  convert_element_type_715 = None
        convert_element_type_718: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_622, torch.float32);  view_622 = None
        view_623: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(arg588_1, _shape_param_627);  arg588_1 = _shape_param_627 = None
        convert_element_type_719: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_623, torch.float32);  view_623 = None
        mul_572: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_719, 0.7071067811865476)
        erf_22: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_572);  mul_572 = None
        add_129: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
        mul_573: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(add_129, 0.5);  add_129 = None
        mul_574: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_719, convert_element_type_719)
        mul_575: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_574, -0.5);  mul_574 = None
        exp_44: "f32[8, 512, 6144]" = torch.ops.aten.exp.default(mul_575);  mul_575 = None
        mul_576: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_44, 0.3989422804014327);  exp_44 = None
        mul_577: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_719, mul_576);  convert_element_type_719 = mul_576 = None
        add_130: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(mul_573, mul_577);  mul_573 = mul_577 = None
        mul_578: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_718, add_130);  convert_element_type_718 = add_130 = None
        convert_element_type_720: "bf16[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(mul_578, torch.bfloat16);  mul_578 = None
        view_624: "bf16[4096, 6144]" = torch.ops.aten.view.default(convert_element_type_720, _shape_param_628);  convert_element_type_720 = _shape_param_628 = None
        mm_258: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_624, arg589_1);  arg589_1 = None
        permute_234: "bf16[6144, 4096]" = torch.ops.aten.permute.default(view_624, [1, 0])
        mm_259: "bf16[6144, 1536]" = torch.ops.aten.mm.default(permute_234, arg590_1);  permute_234 = arg590_1 = None
        sum_328: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_624, [0], True, dtype = torch.float32);  view_624 = None
        view_625: "f32[6144]" = torch.ops.aten.view.default(sum_328, _shape_param_629);  sum_328 = _shape_param_629 = None
        convert_element_type_721: "bf16[6144]" = torch.ops.prims.convert_element_type.default(view_625, torch.bfloat16);  view_625 = None
        view_626: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_258, _shape_param_630);  mm_258 = _shape_param_630 = None
        convert_element_type_722: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_626, torch.float32);  view_626 = None
        add_131: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_568, convert_element_type_722);  mul_568 = convert_element_type_722 = None
        convert_element_type_723: "f32[6144, 1536]" = torch.ops.prims.convert_element_type.default(mm_259, torch.float32);  mm_259 = None
        convert_element_type_724: "f32[6144]" = torch.ops.prims.convert_element_type.default(convert_element_type_721, torch.float32);  convert_element_type_721 = None
        mul_579: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_131, arg591_1);  arg591_1 = None
        mul_580: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_579, 1536)
        sum_329: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_579, [2], True)
        mul_581: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_579, arg592_1);  mul_579 = None
        sum_330: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_581, [2], True);  mul_581 = None
        mul_582: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg592_1, sum_330);  sum_330 = None
        sub_113: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_580, sum_329);  mul_580 = sum_329 = None
        sub_114: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_113, mul_582);  sub_113 = mul_582 = None
        mul_583: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg593_1, sub_114);  arg593_1 = sub_114 = None
        mul_584: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_131, arg592_1);  arg592_1 = None
        sum_331: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_584, [0, 1]);  mul_584 = None
        sum_332: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_131, [0, 1]);  add_131 = None
        convert_element_type_725: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(mul_583, torch.bfloat16)
        convert_element_type_726: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(arg594_1, torch.bfloat16);  arg594_1 = None
        mul_585: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_726, 1.1111111111111112);  convert_element_type_726 = None
        mul_586: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_725, mul_585);  convert_element_type_725 = mul_585 = None
        view_627: "bf16[4096, 1536]" = torch.ops.aten.view.default(mul_586, _shape_param_631);  mul_586 = _shape_param_631 = None
        mm_260: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_627, arg595_1);  arg595_1 = None
        permute_235: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_627, [1, 0])
        mm_261: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_235, arg596_1);  permute_235 = arg596_1 = None
        sum_333: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_627, [0], True, dtype = torch.float32);  view_627 = None
        view_628: "f32[1536]" = torch.ops.aten.view.default(sum_333, _shape_param_632);  sum_333 = _shape_param_632 = None
        convert_element_type_727: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_628, torch.bfloat16);  view_628 = None
        view_629: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_260, _shape_param_633);  mm_260 = _shape_param_633 = None
        convert_element_type_728: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_261, torch.float32);  mm_261 = None
        convert_element_type_729: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_727, torch.float32);  convert_element_type_727 = None
        view_630: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_629, _shape_param_634);  view_629 = _shape_param_634 = None
        permute_236: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_630, [0, 2, 1, 3]);  view_630 = None
        clone_84: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_236, memory_format = torch.contiguous_format);  permute_236 = None
        view_631: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_84, _shape_param_635);  clone_84 = _shape_param_635 = None
        bmm_84: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(arg597_1, view_631);  arg597_1 = None
        bmm_85: "bf16[192, 512, 512]" = torch.ops.aten.bmm.default(view_631, arg598_1);  view_631 = arg598_1 = None
        convert_element_type_730: "f32[192, 512, 512]" = torch.ops.prims.convert_element_type.default(bmm_85, torch.float32);  bmm_85 = None
        view_632: "f32[8, 24, 512, 512]" = torch.ops.aten.view.default(convert_element_type_730, _shape_param_636);  convert_element_type_730 = _shape_param_636 = None
        convert_element_type_731: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(arg599_1, torch.float32);  arg599_1 = None
        mul_587: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_731, 1.1111111111111112);  convert_element_type_731 = None
        mul_588: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(view_632, mul_587);  view_632 = mul_587 = None
        convert_element_type_732: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(arg600_1, torch.float32);  arg600_1 = None
        sub_115: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_732, arg601_1);  convert_element_type_732 = arg601_1 = None
        exp_45: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_115);  sub_115 = None
        div_44: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_45, arg602_1);  exp_45 = arg602_1 = None
        mul_589: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_588, div_44);  mul_588 = None
        sum_334: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_589, [-1], True)
        neg_21: "f32[8, 24, 512, 512]" = torch.ops.aten.neg.default(div_44);  div_44 = None
        fma_21: "f32[8, 24, 512, 512]" = torch.ops.prims.fma.default(neg_21, sum_334, mul_589);  neg_21 = sum_334 = mul_589 = None
        convert_element_type_733: "bf16[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(fma_21, torch.bfloat16);  fma_21 = None
        where_24: "bf16[8, 24, 512, 512]" = torch.ops.aten.where.self(full_3, full_2, convert_element_type_733);  convert_element_type_733 = None
        view_633: "bf16[192, 512, 512]" = torch.ops.aten.view.default(where_24, _shape_param_637);  where_24 = _shape_param_637 = None
        bmm_86: "bf16[192, 64, 512]" = torch.ops.aten.bmm.default(arg603_1, view_633);  arg603_1 = None
        bmm_87: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(view_633, arg604_1);  view_633 = arg604_1 = None
        div_45: "bf16[192, 64, 512]" = torch.ops.aten.div.Tensor(bmm_86, full_4);  bmm_86 = None
        permute_237: "bf16[192, 512, 64]" = torch.ops.aten.permute.default(div_45, [0, 2, 1]);  div_45 = None
        view_634: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_84, _shape_param_638);  bmm_84 = _shape_param_638 = None
        permute_238: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_634, [0, 2, 1, 3]);  view_634 = None
        clone_85: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_238, memory_format = torch.contiguous_format);  permute_238 = None
        view_635: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_85, _shape_param_639);  clone_85 = _shape_param_639 = None
        view_636: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_635, _shape_param_640);  view_635 = _shape_param_640 = None
        mm_262: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_636, arg605_1);  arg605_1 = None
        permute_239: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_636, [1, 0])
        mm_263: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_239, arg606_1);  permute_239 = None
        sum_335: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_636, [0], True, dtype = torch.float32);  view_636 = None
        view_637: "f32[1536]" = torch.ops.aten.view.default(sum_335, _shape_param_641);  sum_335 = _shape_param_641 = None
        convert_element_type_734: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_637, torch.bfloat16);  view_637 = None
        view_638: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_262, _shape_param_642);  mm_262 = _shape_param_642 = None
        convert_element_type_735: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_638, torch.float32);  view_638 = None
        add_132: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_583, convert_element_type_735);  mul_583 = convert_element_type_735 = None
        convert_element_type_736: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_263, torch.float32);  mm_263 = None
        convert_element_type_737: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_734, torch.float32);  convert_element_type_734 = None
        view_639: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(permute_237, _shape_param_643);  permute_237 = _shape_param_643 = None
        permute_240: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_639, [0, 2, 1, 3]);  view_639 = None
        view_640: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(permute_240, _shape_param_644);  permute_240 = _shape_param_644 = None
        clone_86: "bf16[8, 512, 1536]" = torch.ops.aten.clone.default(view_640, memory_format = torch.contiguous_format);  view_640 = None
        view_641: "bf16[4096, 1536]" = torch.ops.aten.view.default(clone_86, _shape_param_645);  clone_86 = _shape_param_645 = None
        mm_264: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_641, arg607_1);  arg607_1 = None
        permute_241: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_641, [1, 0])
        mm_265: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_241, arg606_1);  permute_241 = None
        sum_336: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_641, [0], True, dtype = torch.float32);  view_641 = None
        view_642: "f32[1536]" = torch.ops.aten.view.default(sum_336, _shape_param_646);  sum_336 = _shape_param_646 = None
        convert_element_type_738: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_642, torch.bfloat16);  view_642 = None
        view_643: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_264, _shape_param_647);  mm_264 = _shape_param_647 = None
        convert_element_type_739: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_643, torch.float32);  view_643 = None
        add_133: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_132, convert_element_type_739);  add_132 = convert_element_type_739 = None
        convert_element_type_740: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_265, torch.float32);  mm_265 = None
        convert_element_type_741: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_738, torch.float32);  convert_element_type_738 = None
        view_644: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_87, _shape_param_648);  bmm_87 = _shape_param_648 = None
        permute_242: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_644, [0, 2, 1, 3]);  view_644 = None
        clone_87: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_242, memory_format = torch.contiguous_format);  permute_242 = None
        view_645: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_87, _shape_param_649);  clone_87 = _shape_param_649 = None
        view_646: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_645, _shape_param_650);  view_645 = _shape_param_650 = None
        mm_266: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_646, arg608_1);  arg608_1 = None
        permute_243: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_646, [1, 0])
        mm_267: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_243, arg606_1);  permute_243 = arg606_1 = None
        sum_337: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_646, [0], True, dtype = torch.float32);  view_646 = None
        view_647: "f32[1536]" = torch.ops.aten.view.default(sum_337, _shape_param_651);  sum_337 = _shape_param_651 = None
        convert_element_type_742: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_647, torch.bfloat16);  view_647 = None
        view_648: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_266, _shape_param_652);  mm_266 = _shape_param_652 = None
        convert_element_type_743: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_648, torch.float32);  view_648 = None
        add_134: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_133, convert_element_type_743);  add_133 = convert_element_type_743 = None
        convert_element_type_744: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_267, torch.float32);  mm_267 = None
        convert_element_type_745: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_742, torch.float32);  convert_element_type_742 = None
        mul_590: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_134, arg609_1);  arg609_1 = None
        mul_591: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_590, 1536)
        sum_338: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_590, [2], True)
        mul_592: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_590, arg610_1);  mul_590 = None
        sum_339: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_592, [2], True);  mul_592 = None
        mul_593: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg610_1, sum_339);  sum_339 = None
        sub_116: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_591, sum_338);  mul_591 = sum_338 = None
        sub_117: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_116, mul_593);  sub_116 = mul_593 = None
        mul_594: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg611_1, sub_117);  arg611_1 = sub_117 = None
        mul_595: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_134, arg610_1);  arg610_1 = None
        sum_340: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_595, [0, 1]);  mul_595 = None
        sum_341: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_134, [0, 1]);  add_134 = None
        convert_element_type_746: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(mul_594, torch.bfloat16)
        convert_element_type_747: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(arg612_1, torch.bfloat16);  arg612_1 = None
        mul_596: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_747, 1.1111111111111112);  convert_element_type_747 = None
        mul_597: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_746, mul_596);  convert_element_type_746 = mul_596 = None
        view_649: "bf16[4096, 1536]" = torch.ops.aten.view.default(mul_597, _shape_param_653);  mul_597 = _shape_param_653 = None
        mm_268: "bf16[4096, 6144]" = torch.ops.aten.mm.default(view_649, arg613_1);  arg613_1 = None
        permute_244: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_649, [1, 0])
        mm_269: "bf16[1536, 6144]" = torch.ops.aten.mm.default(permute_244, arg614_1);  permute_244 = arg614_1 = None
        sum_342: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_649, [0], True, dtype = torch.float32);  view_649 = None
        view_650: "f32[1536]" = torch.ops.aten.view.default(sum_342, _shape_param_654);  sum_342 = _shape_param_654 = None
        convert_element_type_748: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_650, torch.bfloat16);  view_650 = None
        view_651: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(mm_268, _shape_param_655);  mm_268 = _shape_param_655 = None
        convert_element_type_749: "f32[1536, 6144]" = torch.ops.prims.convert_element_type.default(mm_269, torch.float32);  mm_269 = None
        convert_element_type_750: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_748, torch.float32);  convert_element_type_748 = None
        convert_element_type_751: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_651, torch.float32);  view_651 = None
        view_652: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(arg615_1, _shape_param_656);  arg615_1 = _shape_param_656 = None
        convert_element_type_752: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_652, torch.float32);  view_652 = None
        mul_598: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_752, 0.7071067811865476)
        erf_23: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_598);  mul_598 = None
        add_135: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
        mul_599: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(add_135, 0.5);  add_135 = None
        mul_600: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_752, convert_element_type_752)
        mul_601: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_600, -0.5);  mul_600 = None
        exp_46: "f32[8, 512, 6144]" = torch.ops.aten.exp.default(mul_601);  mul_601 = None
        mul_602: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_46, 0.3989422804014327);  exp_46 = None
        mul_603: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_752, mul_602);  convert_element_type_752 = mul_602 = None
        add_136: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(mul_599, mul_603);  mul_599 = mul_603 = None
        mul_604: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_751, add_136);  convert_element_type_751 = add_136 = None
        convert_element_type_753: "bf16[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(mul_604, torch.bfloat16);  mul_604 = None
        view_653: "bf16[4096, 6144]" = torch.ops.aten.view.default(convert_element_type_753, _shape_param_657);  convert_element_type_753 = _shape_param_657 = None
        mm_270: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_653, arg616_1);  arg616_1 = None
        permute_245: "bf16[6144, 4096]" = torch.ops.aten.permute.default(view_653, [1, 0])
        mm_271: "bf16[6144, 1536]" = torch.ops.aten.mm.default(permute_245, arg617_1);  permute_245 = arg617_1 = None
        sum_343: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_653, [0], True, dtype = torch.float32);  view_653 = None
        view_654: "f32[6144]" = torch.ops.aten.view.default(sum_343, _shape_param_658);  sum_343 = _shape_param_658 = None
        convert_element_type_754: "bf16[6144]" = torch.ops.prims.convert_element_type.default(view_654, torch.bfloat16);  view_654 = None
        view_655: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_270, _shape_param_659);  mm_270 = _shape_param_659 = None
        convert_element_type_755: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_655, torch.float32);  view_655 = None
        add_137: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_594, convert_element_type_755);  mul_594 = convert_element_type_755 = None
        convert_element_type_756: "f32[6144, 1536]" = torch.ops.prims.convert_element_type.default(mm_271, torch.float32);  mm_271 = None
        convert_element_type_757: "f32[6144]" = torch.ops.prims.convert_element_type.default(convert_element_type_754, torch.float32);  convert_element_type_754 = None
        mul_605: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_137, arg618_1);  arg618_1 = None
        mul_606: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_605, 1536)
        sum_344: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_605, [2], True)
        mul_607: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_605, arg619_1);  mul_605 = None
        sum_345: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_607, [2], True);  mul_607 = None
        mul_608: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg619_1, sum_345);  sum_345 = None
        sub_118: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_606, sum_344);  mul_606 = sum_344 = None
        sub_119: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_118, mul_608);  sub_118 = mul_608 = None
        mul_609: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg620_1, sub_119);  arg620_1 = sub_119 = None
        mul_610: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_137, arg619_1);  arg619_1 = None
        sum_346: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_610, [0, 1]);  mul_610 = None
        sum_347: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_137, [0, 1]);  add_137 = None
        convert_element_type_758: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(mul_609, torch.bfloat16)
        convert_element_type_759: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(arg621_1, torch.bfloat16);  arg621_1 = None
        mul_611: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_759, 1.1111111111111112);  convert_element_type_759 = None
        mul_612: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_758, mul_611);  convert_element_type_758 = mul_611 = None
        view_656: "bf16[4096, 1536]" = torch.ops.aten.view.default(mul_612, _shape_param_660);  mul_612 = _shape_param_660 = None
        mm_272: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_656, arg622_1);  arg622_1 = None
        permute_246: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_656, [1, 0])
        mm_273: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_246, arg623_1);  permute_246 = arg623_1 = None
        sum_348: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_656, [0], True, dtype = torch.float32);  view_656 = None
        view_657: "f32[1536]" = torch.ops.aten.view.default(sum_348, _shape_param_661);  sum_348 = _shape_param_661 = None
        convert_element_type_760: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_657, torch.bfloat16);  view_657 = None
        view_658: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_272, _shape_param_662);  mm_272 = _shape_param_662 = None
        convert_element_type_761: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_273, torch.float32);  mm_273 = None
        convert_element_type_762: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_760, torch.float32);  convert_element_type_760 = None
        view_659: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_658, _shape_param_663);  view_658 = _shape_param_663 = None
        permute_247: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_659, [0, 2, 1, 3]);  view_659 = None
        clone_88: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_247, memory_format = torch.contiguous_format);  permute_247 = None
        view_660: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_88, _shape_param_664);  clone_88 = _shape_param_664 = None
        bmm_88: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(arg624_1, view_660);  arg624_1 = None
        bmm_89: "bf16[192, 512, 512]" = torch.ops.aten.bmm.default(view_660, arg625_1);  view_660 = arg625_1 = None
        convert_element_type_763: "f32[192, 512, 512]" = torch.ops.prims.convert_element_type.default(bmm_89, torch.float32);  bmm_89 = None
        view_661: "f32[8, 24, 512, 512]" = torch.ops.aten.view.default(convert_element_type_763, _shape_param_665);  convert_element_type_763 = _shape_param_665 = None
        convert_element_type_764: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(arg626_1, torch.float32);  arg626_1 = None
        mul_613: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_764, 1.1111111111111112);  convert_element_type_764 = None
        mul_614: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(view_661, mul_613);  view_661 = mul_613 = None
        convert_element_type_765: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(arg627_1, torch.float32);  arg627_1 = None
        sub_120: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_765, arg628_1);  convert_element_type_765 = arg628_1 = None
        exp_47: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_120);  sub_120 = None
        div_46: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_47, arg629_1);  exp_47 = arg629_1 = None
        mul_615: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_614, div_46);  mul_614 = None
        sum_349: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_615, [-1], True)
        neg_22: "f32[8, 24, 512, 512]" = torch.ops.aten.neg.default(div_46);  div_46 = None
        fma_22: "f32[8, 24, 512, 512]" = torch.ops.prims.fma.default(neg_22, sum_349, mul_615);  neg_22 = sum_349 = mul_615 = None
        convert_element_type_766: "bf16[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(fma_22, torch.bfloat16);  fma_22 = None
        where_25: "bf16[8, 24, 512, 512]" = torch.ops.aten.where.self(full_3, full_2, convert_element_type_766);  convert_element_type_766 = None
        view_662: "bf16[192, 512, 512]" = torch.ops.aten.view.default(where_25, _shape_param_666);  where_25 = _shape_param_666 = None
        bmm_90: "bf16[192, 64, 512]" = torch.ops.aten.bmm.default(arg630_1, view_662);  arg630_1 = None
        bmm_91: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(view_662, arg631_1);  view_662 = arg631_1 = None
        div_47: "bf16[192, 64, 512]" = torch.ops.aten.div.Tensor(bmm_90, full_4);  bmm_90 = None
        permute_248: "bf16[192, 512, 64]" = torch.ops.aten.permute.default(div_47, [0, 2, 1]);  div_47 = None
        view_663: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_88, _shape_param_667);  bmm_88 = _shape_param_667 = None
        permute_249: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_663, [0, 2, 1, 3]);  view_663 = None
        clone_89: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_249, memory_format = torch.contiguous_format);  permute_249 = None
        view_664: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_89, _shape_param_668);  clone_89 = _shape_param_668 = None
        view_665: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_664, _shape_param_669);  view_664 = _shape_param_669 = None
        mm_274: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_665, arg632_1);  arg632_1 = None
        permute_250: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_665, [1, 0])
        mm_275: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_250, arg633_1);  permute_250 = None
        sum_350: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_665, [0], True, dtype = torch.float32);  view_665 = None
        view_666: "f32[1536]" = torch.ops.aten.view.default(sum_350, _shape_param_670);  sum_350 = _shape_param_670 = None
        convert_element_type_767: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_666, torch.bfloat16);  view_666 = None
        view_667: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_274, _shape_param_671);  mm_274 = _shape_param_671 = None
        convert_element_type_768: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_667, torch.float32);  view_667 = None
        add_138: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_609, convert_element_type_768);  mul_609 = convert_element_type_768 = None
        convert_element_type_769: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_275, torch.float32);  mm_275 = None
        convert_element_type_770: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_767, torch.float32);  convert_element_type_767 = None
        view_668: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(permute_248, _shape_param_672);  permute_248 = _shape_param_672 = None
        permute_251: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_668, [0, 2, 1, 3]);  view_668 = None
        view_669: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(permute_251, _shape_param_673);  permute_251 = _shape_param_673 = None
        clone_90: "bf16[8, 512, 1536]" = torch.ops.aten.clone.default(view_669, memory_format = torch.contiguous_format);  view_669 = None
        view_670: "bf16[4096, 1536]" = torch.ops.aten.view.default(clone_90, _shape_param_674);  clone_90 = _shape_param_674 = None
        mm_276: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_670, arg634_1);  arg634_1 = None
        permute_252: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_670, [1, 0])
        mm_277: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_252, arg633_1);  permute_252 = None
        sum_351: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_670, [0], True, dtype = torch.float32);  view_670 = None
        view_671: "f32[1536]" = torch.ops.aten.view.default(sum_351, _shape_param_675);  sum_351 = _shape_param_675 = None
        convert_element_type_771: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_671, torch.bfloat16);  view_671 = None
        view_672: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_276, _shape_param_676);  mm_276 = _shape_param_676 = None
        convert_element_type_772: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_672, torch.float32);  view_672 = None
        add_139: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_138, convert_element_type_772);  add_138 = convert_element_type_772 = None
        convert_element_type_773: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_277, torch.float32);  mm_277 = None
        convert_element_type_774: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_771, torch.float32);  convert_element_type_771 = None
        view_673: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_91, _shape_param_677);  bmm_91 = _shape_param_677 = None
        permute_253: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_673, [0, 2, 1, 3]);  view_673 = None
        clone_91: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_253, memory_format = torch.contiguous_format);  permute_253 = None
        view_674: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_91, _shape_param_678);  clone_91 = _shape_param_678 = None
        view_675: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_674, _shape_param_679);  view_674 = _shape_param_679 = None
        mm_278: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_675, arg635_1);  arg635_1 = None
        permute_254: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_675, [1, 0])
        mm_279: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_254, arg633_1);  permute_254 = arg633_1 = None
        sum_352: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_675, [0], True, dtype = torch.float32);  view_675 = None
        view_676: "f32[1536]" = torch.ops.aten.view.default(sum_352, _shape_param_680);  sum_352 = _shape_param_680 = None
        convert_element_type_775: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_676, torch.bfloat16);  view_676 = None
        view_677: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_278, _shape_param_681);  mm_278 = _shape_param_681 = None
        convert_element_type_776: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_677, torch.float32);  view_677 = None
        add_140: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_139, convert_element_type_776);  add_139 = convert_element_type_776 = None
        convert_element_type_777: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_279, torch.float32);  mm_279 = None
        convert_element_type_778: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_775, torch.float32);  convert_element_type_775 = None
        mul_616: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_140, arg636_1);  arg636_1 = None
        mul_617: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_616, 1536)
        sum_353: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_616, [2], True)
        mul_618: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_616, arg637_1);  mul_616 = None
        sum_354: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_618, [2], True);  mul_618 = None
        mul_619: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg637_1, sum_354);  sum_354 = None
        sub_121: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_617, sum_353);  mul_617 = sum_353 = None
        sub_122: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_121, mul_619);  sub_121 = mul_619 = None
        mul_620: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg638_1, sub_122);  arg638_1 = sub_122 = None
        mul_621: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_140, arg637_1);  arg637_1 = None
        sum_355: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_621, [0, 1]);  mul_621 = None
        sum_356: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_140, [0, 1]);  add_140 = None
        convert_element_type_779: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(mul_620, torch.bfloat16)
        convert_element_type_780: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(arg639_1, torch.bfloat16);  arg639_1 = None
        mul_622: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_780, 1.1111111111111112);  convert_element_type_780 = None
        mul_623: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_779, mul_622);  convert_element_type_779 = mul_622 = None
        view_678: "bf16[4096, 1536]" = torch.ops.aten.view.default(mul_623, _shape_param_682);  mul_623 = _shape_param_682 = None
        mm_280: "bf16[4096, 6144]" = torch.ops.aten.mm.default(view_678, arg640_1);  arg640_1 = None
        permute_255: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_678, [1, 0])
        mm_281: "bf16[1536, 6144]" = torch.ops.aten.mm.default(permute_255, arg641_1);  permute_255 = arg641_1 = None
        sum_357: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_678, [0], True, dtype = torch.float32);  view_678 = None
        view_679: "f32[1536]" = torch.ops.aten.view.default(sum_357, _shape_param_683);  sum_357 = _shape_param_683 = None
        convert_element_type_781: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_679, torch.bfloat16);  view_679 = None
        view_680: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(mm_280, _shape_param_684);  mm_280 = _shape_param_684 = None
        convert_element_type_782: "f32[1536, 6144]" = torch.ops.prims.convert_element_type.default(mm_281, torch.float32);  mm_281 = None
        convert_element_type_783: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_781, torch.float32);  convert_element_type_781 = None
        convert_element_type_784: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_680, torch.float32);  view_680 = None
        view_681: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(arg642_1, _shape_param_685);  arg642_1 = _shape_param_685 = None
        convert_element_type_785: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_681, torch.float32);  view_681 = None
        mul_624: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_785, 0.7071067811865476)
        erf_24: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_624);  mul_624 = None
        add_141: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_24, 1);  erf_24 = None
        mul_625: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(add_141, 0.5);  add_141 = None
        mul_626: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_785, convert_element_type_785)
        mul_627: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_626, -0.5);  mul_626 = None
        exp_48: "f32[8, 512, 6144]" = torch.ops.aten.exp.default(mul_627);  mul_627 = None
        mul_628: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_48, 0.3989422804014327);  exp_48 = None
        mul_629: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_785, mul_628);  convert_element_type_785 = mul_628 = None
        add_142: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(mul_625, mul_629);  mul_625 = mul_629 = None
        mul_630: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_784, add_142);  convert_element_type_784 = add_142 = None
        convert_element_type_786: "bf16[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(mul_630, torch.bfloat16);  mul_630 = None
        view_682: "bf16[4096, 6144]" = torch.ops.aten.view.default(convert_element_type_786, _shape_param_686);  convert_element_type_786 = _shape_param_686 = None
        mm_282: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_682, arg643_1);  arg643_1 = None
        permute_256: "bf16[6144, 4096]" = torch.ops.aten.permute.default(view_682, [1, 0])
        mm_283: "bf16[6144, 1536]" = torch.ops.aten.mm.default(permute_256, arg644_1);  permute_256 = arg644_1 = None
        sum_358: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_682, [0], True, dtype = torch.float32);  view_682 = None
        view_683: "f32[6144]" = torch.ops.aten.view.default(sum_358, _shape_param_687);  sum_358 = _shape_param_687 = None
        convert_element_type_787: "bf16[6144]" = torch.ops.prims.convert_element_type.default(view_683, torch.bfloat16);  view_683 = None
        view_684: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_282, _shape_param_688);  mm_282 = _shape_param_688 = None
        convert_element_type_788: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_684, torch.float32);  view_684 = None
        add_143: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_620, convert_element_type_788);  mul_620 = convert_element_type_788 = None
        convert_element_type_789: "f32[6144, 1536]" = torch.ops.prims.convert_element_type.default(mm_283, torch.float32);  mm_283 = None
        convert_element_type_790: "f32[6144]" = torch.ops.prims.convert_element_type.default(convert_element_type_787, torch.float32);  convert_element_type_787 = None
        mul_631: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_143, arg645_1);  arg645_1 = None
        mul_632: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_631, 1536)
        sum_359: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_631, [2], True)
        mul_633: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_631, arg646_1);  mul_631 = None
        sum_360: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_633, [2], True);  mul_633 = None
        mul_634: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg646_1, sum_360);  sum_360 = None
        sub_123: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_632, sum_359);  mul_632 = sum_359 = None
        sub_124: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_123, mul_634);  sub_123 = mul_634 = None
        mul_635: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(arg647_1, sub_124);  arg647_1 = sub_124 = None
        mul_636: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_143, arg646_1);  arg646_1 = None
        sum_361: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_636, [0, 1]);  mul_636 = None
        sum_362: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_143, [0, 1]);  add_143 = None
        convert_element_type_791: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(mul_635, torch.bfloat16)
        convert_element_type_792: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(arg648_1, torch.bfloat16);  arg648_1 = None
        mul_637: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_792, 1.1111111111111112);  convert_element_type_792 = None
        mul_638: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_791, mul_637);  convert_element_type_791 = mul_637 = None
        view_685: "bf16[4096, 1536]" = torch.ops.aten.view.default(mul_638, _shape_param_689);  mul_638 = _shape_param_689 = None
        mm_284: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_685, arg649_1);  arg649_1 = None
        permute_257: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_685, [1, 0])
        mm_285: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_257, arg650_1);  permute_257 = arg650_1 = None
        sum_363: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_685, [0], True, dtype = torch.float32);  view_685 = None
        view_686: "f32[1536]" = torch.ops.aten.view.default(sum_363, _shape_param_690);  sum_363 = _shape_param_690 = None
        convert_element_type_793: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_686, torch.bfloat16);  view_686 = None
        view_687: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_284, _shape_param_691);  mm_284 = _shape_param_691 = None
        convert_element_type_794: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_285, torch.float32);  mm_285 = None
        convert_element_type_795: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_793, torch.float32);  convert_element_type_793 = None
        view_688: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_687, _shape_param_692);  view_687 = _shape_param_692 = None
        permute_258: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_688, [0, 2, 1, 3]);  view_688 = None
        clone_92: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_258, memory_format = torch.contiguous_format);  permute_258 = None
        view_689: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_92, _shape_param_693);  clone_92 = _shape_param_693 = None
        bmm_92: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(arg651_1, view_689);  arg651_1 = None
        bmm_93: "bf16[192, 512, 512]" = torch.ops.aten.bmm.default(view_689, arg652_1);  view_689 = arg652_1 = None
        convert_element_type_796: "f32[192, 512, 512]" = torch.ops.prims.convert_element_type.default(bmm_93, torch.float32);  bmm_93 = None
        view_690: "f32[8, 24, 512, 512]" = torch.ops.aten.view.default(convert_element_type_796, _shape_param_694);  convert_element_type_796 = _shape_param_694 = None
        convert_element_type_797: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(arg653_1, torch.float32);  arg653_1 = None
        mul_639: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_797, 1.1111111111111112);  convert_element_type_797 = None
        mul_640: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(view_690, mul_639);  view_690 = mul_639 = None
        view_691: "bf16[8, 24, 512, 512]" = torch.ops.aten.view.default(arg654_1, _shape_param_695);  arg654_1 = _shape_param_695 = None
        full_5: "bf16[]" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_26: "bf16[8, 24, 512, 512]" = torch.ops.aten.where.self(full_3, full_5, view_691);  full_5 = view_691 = None
        convert_element_type_798: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(where_26, torch.float32);  where_26 = None
        sub_125: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_798, arg655_1);  convert_element_type_798 = arg655_1 = None
        exp_49: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_125);  sub_125 = None
        div_48: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_49, arg656_1);  exp_49 = arg656_1 = None
        mul_641: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_640, div_48);  mul_640 = None
        sum_364: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_641, [-1], True)
        neg_23: "f32[8, 24, 512, 512]" = torch.ops.aten.neg.default(div_48);  div_48 = None
        fma_23: "f32[8, 24, 512, 512]" = torch.ops.prims.fma.default(neg_23, sum_364, mul_641);  neg_23 = sum_364 = mul_641 = None
        convert_element_type_799: "bf16[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(fma_23, torch.bfloat16);  fma_23 = None
        where_27: "bf16[8, 24, 512, 512]" = torch.ops.aten.where.self(full_3, full_2, convert_element_type_799);  full_3 = full_2 = convert_element_type_799 = None
        view_692: "bf16[192, 512, 512]" = torch.ops.aten.view.default(where_27, _shape_param_696);  where_27 = _shape_param_696 = None
        bmm_94: "bf16[192, 64, 512]" = torch.ops.aten.bmm.default(arg657_1, view_692);  arg657_1 = None
        bmm_95: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(view_692, arg658_1);  view_692 = arg658_1 = None
        div_49: "bf16[192, 64, 512]" = torch.ops.aten.div.Tensor(bmm_94, full_4);  bmm_94 = full_4 = None
        permute_259: "bf16[192, 512, 64]" = torch.ops.aten.permute.default(div_49, [0, 2, 1]);  div_49 = None
        view_693: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_92, _shape_param_697);  bmm_92 = _shape_param_697 = None
        permute_260: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_693, [0, 2, 1, 3]);  view_693 = None
        clone_93: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_260, memory_format = torch.contiguous_format);  permute_260 = None
        view_694: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_93, _shape_param_698);  clone_93 = _shape_param_698 = None
        view_695: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_694, _shape_param_699);  view_694 = _shape_param_699 = None
        mm_286: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_695, arg659_1);  arg659_1 = None
        permute_261: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_695, [1, 0])
        mm_287: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_261, arg660_1);  permute_261 = None
        sum_365: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_695, [0], True, dtype = torch.float32);  view_695 = None
        view_696: "f32[1536]" = torch.ops.aten.view.default(sum_365, _shape_param_700);  sum_365 = _shape_param_700 = None
        convert_element_type_800: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_696, torch.bfloat16);  view_696 = None
        view_697: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_286, _shape_param_701);  mm_286 = _shape_param_701 = None
        convert_element_type_801: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_697, torch.float32);  view_697 = None
        add_144: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_635, convert_element_type_801);  mul_635 = convert_element_type_801 = None
        convert_element_type_802: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_287, torch.float32);  mm_287 = None
        convert_element_type_803: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_800, torch.float32);  convert_element_type_800 = None
        view_698: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(permute_259, _shape_param_702);  permute_259 = _shape_param_702 = None
        permute_262: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_698, [0, 2, 1, 3]);  view_698 = None
        view_699: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(permute_262, _shape_param_703);  permute_262 = _shape_param_703 = None
        clone_94: "bf16[8, 512, 1536]" = torch.ops.aten.clone.default(view_699, memory_format = torch.contiguous_format);  view_699 = None
        view_700: "bf16[4096, 1536]" = torch.ops.aten.view.default(clone_94, _shape_param_704);  clone_94 = _shape_param_704 = None
        mm_288: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_700, arg661_1);  arg661_1 = None
        permute_263: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_700, [1, 0])
        mm_289: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_263, arg660_1);  permute_263 = None
        sum_366: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_700, [0], True, dtype = torch.float32);  view_700 = None
        view_701: "f32[1536]" = torch.ops.aten.view.default(sum_366, _shape_param_705);  sum_366 = _shape_param_705 = None
        convert_element_type_804: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_701, torch.bfloat16);  view_701 = None
        view_702: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_288, _shape_param_706);  mm_288 = _shape_param_706 = None
        convert_element_type_805: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_702, torch.float32);  view_702 = None
        add_145: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_144, convert_element_type_805);  add_144 = convert_element_type_805 = None
        convert_element_type_806: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_289, torch.float32);  mm_289 = None
        convert_element_type_807: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_804, torch.float32);  convert_element_type_804 = None
        view_703: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_95, _shape_param_707);  bmm_95 = _shape_param_707 = None
        permute_264: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_703, [0, 2, 1, 3]);  view_703 = None
        clone_95: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_264, memory_format = torch.contiguous_format);  permute_264 = None
        view_704: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_95, _shape_param_708);  clone_95 = _shape_param_708 = None
        view_705: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_704, _shape_param_709);  view_704 = _shape_param_709 = None
        mm_290: "bf16[4096, 1536]" = torch.ops.aten.mm.default(view_705, arg662_1);  arg662_1 = None
        permute_265: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_705, [1, 0])
        mm_291: "bf16[1536, 1536]" = torch.ops.aten.mm.default(permute_265, arg660_1);  permute_265 = arg660_1 = None
        sum_367: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_705, [0], True, dtype = torch.float32);  view_705 = None
        view_706: "f32[1536]" = torch.ops.aten.view.default(sum_367, _shape_param_710);  sum_367 = _shape_param_710 = None
        convert_element_type_808: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_706, torch.bfloat16);  view_706 = None
        view_707: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(mm_290, _shape_param_711);  mm_290 = _shape_param_711 = None
        convert_element_type_809: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_707, torch.float32);  view_707 = None
        add_146: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_145, convert_element_type_809);  add_145 = convert_element_type_809 = None
        convert_element_type_810: "f32[1536, 1536]" = torch.ops.prims.convert_element_type.default(mm_291, torch.float32);  mm_291 = None
        convert_element_type_811: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_808, torch.float32);  convert_element_type_808 = None
        convert_element_type_812: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(arg663_1, torch.float32);  arg663_1 = None
        mul_642: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_812, 1.1111111111111112);  convert_element_type_812 = None
        mul_643: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_146, mul_642);  add_146 = mul_642 = None
        mul_644: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_643, arg664_1);  arg664_1 = None
        mul_645: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_644, 1536)
        sum_368: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_644, [2], True)
        add_147: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(arg665_1, arg666_1);  arg665_1 = arg666_1 = None
        sub_126: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_147, arg667_1);  add_147 = arg667_1 = None
        mul_646: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_126, arg668_1);  sub_126 = None
        mul_647: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_644, mul_646);  mul_644 = None
        sum_369: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_647, [2], True);  mul_647 = None
        mul_648: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_646, sum_369);  sum_369 = None
        sub_127: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_645, sum_368);  mul_645 = sum_368 = None
        sub_128: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_127, mul_648);  sub_127 = mul_648 = None
        div_50: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(arg668_1, 1536);  arg668_1 = None
        mul_649: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(div_50, sub_128);  div_50 = sub_128 = None
        mul_650: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_643, mul_646);  mul_646 = None
        sum_370: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_650, [0, 1]);  mul_650 = None
        sum_371: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_643, [0, 1]);  mul_643 = None
        sum_372: "f32[1, 512, 1536]" = torch.ops.aten.sum.dim_IntList(mul_649, [0], True, dtype = torch.float32)
        ge: "b8[1, 512]" = torch.ops.aten.ge.Scalar(arg669_1, 0)
        lt: "b8[1, 512]" = torch.ops.aten.lt.Scalar(arg669_1, 512)
        bitwise_and: "b8[1, 512]" = torch.ops.aten.bitwise_and.Tensor(ge, lt);  ge = lt = None
        ne_1: "b8[1, 512]" = torch.ops.aten.ne.Scalar(arg669_1, -1)
        bitwise_and_1: "b8[1, 512]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, ne_1);  bitwise_and = ne_1 = None
        unsqueeze_1: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_1, -1);  bitwise_and_1 = None
        full_6: "f32[512, 1536]" = torch.ops.aten.full.default(_shape_param_712, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_712 = None
        _unsafe_masked_index_put_accumulate: "f32[512, 1536]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_6, unsqueeze_1, [arg669_1], sum_372);  full_6 = unsqueeze_1 = arg669_1 = sum_372 = None
        ge_1: "b8[8, 512]" = torch.ops.aten.ge.Scalar(arg670_1, 0)
        lt_1: "b8[8, 512]" = torch.ops.aten.lt.Scalar(arg670_1, 128100)
        bitwise_and_2: "b8[8, 512]" = torch.ops.aten.bitwise_and.Tensor(ge_1, lt_1);  ge_1 = lt_1 = None
        ne_2: "b8[8, 512]" = torch.ops.aten.ne.Scalar(arg670_1, 0)
        bitwise_and_3: "b8[8, 512]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_2, ne_2);  bitwise_and_2 = ne_2 = None
        unsqueeze_2: "b8[8, 512, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_3, -1);  bitwise_and_3 = None
        full_7: "f32[128100, 1536]" = torch.ops.aten.full.default(_shape_param_713, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_713 = None
        _unsafe_masked_index_put_accumulate_1: "f32[128100, 1536]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_7, unsqueeze_2, [arg670_1], mul_649);  full_7 = unsqueeze_2 = arg670_1 = mul_649 = None
        add_148: "f32[128100, 1536]" = torch.ops.aten.add.Tensor(convert_element_type_8, _unsafe_masked_index_put_accumulate_1);  convert_element_type_8 = _unsafe_masked_index_put_accumulate_1 = None
        return (convert_element_type_9, sum_5, sum_6, convert_element_type_18, convert_element_type_19, sum_10, sum_11, convert_element_type_23, convert_element_type_24, convert_element_type_30, convert_element_type_31, sum_16, sum_17, convert_element_type_35, convert_element_type_36, convert_element_type_43, convert_element_type_44, convert_element_type_47, convert_element_type_48, convert_element_type_51, convert_element_type_52, sum_25, sum_26, convert_element_type_56, convert_element_type_57, convert_element_type_63, convert_element_type_64, sum_31, sum_32, convert_element_type_68, convert_element_type_69, convert_element_type_76, convert_element_type_77, convert_element_type_80, convert_element_type_81, convert_element_type_84, convert_element_type_85, sum_40, sum_41, convert_element_type_89, convert_element_type_90, convert_element_type_96, convert_element_type_97, sum_46, sum_47, convert_element_type_101, convert_element_type_102, convert_element_type_109, convert_element_type_110, convert_element_type_113, convert_element_type_114, convert_element_type_117, convert_element_type_118, sum_55, sum_56, convert_element_type_122, convert_element_type_123, convert_element_type_129, convert_element_type_130, sum_61, sum_62, convert_element_type_134, convert_element_type_135, convert_element_type_142, convert_element_type_143, convert_element_type_146, convert_element_type_147, convert_element_type_150, convert_element_type_151, sum_70, sum_71, convert_element_type_155, convert_element_type_156, convert_element_type_162, convert_element_type_163, sum_76, sum_77, convert_element_type_167, convert_element_type_168, convert_element_type_175, convert_element_type_176, convert_element_type_179, convert_element_type_180, convert_element_type_183, convert_element_type_184, sum_85, sum_86, convert_element_type_188, convert_element_type_189, convert_element_type_195, convert_element_type_196, sum_91, sum_92, convert_element_type_200, convert_element_type_201, convert_element_type_208, convert_element_type_209, convert_element_type_212, convert_element_type_213, convert_element_type_216, convert_element_type_217, sum_100, sum_101, convert_element_type_221, convert_element_type_222, convert_element_type_228, convert_element_type_229, sum_106, sum_107, convert_element_type_233, convert_element_type_234, convert_element_type_241, convert_element_type_242, convert_element_type_245, convert_element_type_246, convert_element_type_249, convert_element_type_250, sum_115, sum_116, convert_element_type_254, convert_element_type_255, convert_element_type_261, convert_element_type_262, sum_121, sum_122, convert_element_type_266, convert_element_type_267, convert_element_type_274, convert_element_type_275, convert_element_type_278, convert_element_type_279, convert_element_type_282, convert_element_type_283, sum_130, sum_131, convert_element_type_287, convert_element_type_288, convert_element_type_294, convert_element_type_295, sum_136, sum_137, convert_element_type_299, convert_element_type_300, convert_element_type_307, convert_element_type_308, convert_element_type_311, convert_element_type_312, convert_element_type_315, convert_element_type_316, sum_145, sum_146, convert_element_type_320, convert_element_type_321, convert_element_type_327, convert_element_type_328, sum_151, sum_152, convert_element_type_332, convert_element_type_333, convert_element_type_340, convert_element_type_341, convert_element_type_344, convert_element_type_345, convert_element_type_348, convert_element_type_349, sum_160, sum_161, convert_element_type_353, convert_element_type_354, convert_element_type_360, convert_element_type_361, sum_166, sum_167, convert_element_type_365, convert_element_type_366, convert_element_type_373, convert_element_type_374, convert_element_type_377, convert_element_type_378, convert_element_type_381, convert_element_type_382, sum_175, sum_176, convert_element_type_386, convert_element_type_387, convert_element_type_393, convert_element_type_394, sum_181, sum_182, convert_element_type_398, convert_element_type_399, convert_element_type_406, convert_element_type_407, convert_element_type_410, convert_element_type_411, convert_element_type_414, convert_element_type_415, sum_190, sum_191, convert_element_type_419, convert_element_type_420, convert_element_type_426, convert_element_type_427, sum_196, sum_197, convert_element_type_431, convert_element_type_432, convert_element_type_439, convert_element_type_440, convert_element_type_443, convert_element_type_444, convert_element_type_447, convert_element_type_448, sum_205, sum_206, convert_element_type_452, convert_element_type_453, convert_element_type_459, convert_element_type_460, sum_211, sum_212, convert_element_type_464, convert_element_type_465, convert_element_type_472, convert_element_type_473, convert_element_type_476, convert_element_type_477, convert_element_type_480, convert_element_type_481, sum_220, sum_221, convert_element_type_485, convert_element_type_486, convert_element_type_492, convert_element_type_493, sum_226, sum_227, convert_element_type_497, convert_element_type_498, convert_element_type_505, convert_element_type_506, convert_element_type_509, convert_element_type_510, convert_element_type_513, convert_element_type_514, sum_235, sum_236, convert_element_type_518, convert_element_type_519, convert_element_type_525, convert_element_type_526, sum_241, sum_242, convert_element_type_530, convert_element_type_531, convert_element_type_538, convert_element_type_539, convert_element_type_542, convert_element_type_543, convert_element_type_546, convert_element_type_547, sum_250, sum_251, convert_element_type_551, convert_element_type_552, convert_element_type_558, convert_element_type_559, sum_256, sum_257, convert_element_type_563, convert_element_type_564, convert_element_type_571, convert_element_type_572, convert_element_type_575, convert_element_type_576, convert_element_type_579, convert_element_type_580, sum_265, sum_266, convert_element_type_584, convert_element_type_585, convert_element_type_591, convert_element_type_592, sum_271, sum_272, convert_element_type_596, convert_element_type_597, convert_element_type_604, convert_element_type_605, convert_element_type_608, convert_element_type_609, convert_element_type_612, convert_element_type_613, sum_280, sum_281, convert_element_type_617, convert_element_type_618, convert_element_type_624, convert_element_type_625, sum_286, sum_287, convert_element_type_629, convert_element_type_630, convert_element_type_637, convert_element_type_638, convert_element_type_641, convert_element_type_642, convert_element_type_645, convert_element_type_646, sum_295, sum_296, convert_element_type_650, convert_element_type_651, convert_element_type_657, convert_element_type_658, sum_301, sum_302, convert_element_type_662, convert_element_type_663, convert_element_type_670, convert_element_type_671, convert_element_type_674, convert_element_type_675, convert_element_type_678, convert_element_type_679, sum_310, sum_311, convert_element_type_683, convert_element_type_684, convert_element_type_690, convert_element_type_691, sum_316, sum_317, convert_element_type_695, convert_element_type_696, convert_element_type_703, convert_element_type_704, convert_element_type_707, convert_element_type_708, convert_element_type_711, convert_element_type_712, sum_325, sum_326, convert_element_type_716, convert_element_type_717, convert_element_type_723, convert_element_type_724, sum_331, sum_332, convert_element_type_728, convert_element_type_729, convert_element_type_736, convert_element_type_737, convert_element_type_740, convert_element_type_741, convert_element_type_744, convert_element_type_745, sum_340, sum_341, convert_element_type_749, convert_element_type_750, convert_element_type_756, convert_element_type_757, sum_346, sum_347, convert_element_type_761, convert_element_type_762, convert_element_type_769, convert_element_type_770, convert_element_type_773, convert_element_type_774, convert_element_type_777, convert_element_type_778, sum_355, sum_356, convert_element_type_782, convert_element_type_783, convert_element_type_789, convert_element_type_790, sum_361, sum_362, convert_element_type_794, convert_element_type_795, convert_element_type_802, convert_element_type_803, convert_element_type_806, convert_element_type_807, convert_element_type_810, convert_element_type_811, sum_370, sum_371, _unsafe_masked_index_put_accumulate, add_148)



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
