"""
Standalone repro captured via capture_hook.
Label: swin_base_patch4_window7_224
Pattern hash: a4c5c8022c8b
Shape hash: a84bf2d8
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
    def forward(self, arg0_1: "bf16[128, 1000]", arg1_1: "bf16[1000, 1024]", arg2_1: "f32[1024]", arg3_1: "bf16[128, 7, 7, 1024]", arg4_1: "f32[128, 7, 7, 1]", arg5_1: "f32[128, 7, 7, 1]", arg6_1: "b8[128, 1, 1]", arg7_1: "bf16[1024, 4096]", arg8_1: "bf16[6272, 4096]", arg9_1: "bf16[6272, 4096]", arg10_1: "bf16[4096, 1024]", arg11_1: "bf16[6272, 1024]", arg12_1: "f32[1024]", arg13_1: "bf16[128, 49, 1024]", arg14_1: "f32[128, 49, 1]", arg15_1: "f32[128, 49, 1]", arg16_1: "b8[128, 1, 1, 1]", arg17_1: "bf16[1024, 1024]", arg18_1: "bf16[6272, 1024]", arg19_1: "bf16[4096, 49, 49]", arg20_1: "bf16[4096, 32, 49]", arg21_1: "bf16[4096, 49, 49]", arg22_1: "i64[49, 49]", arg23_1: "f32[169, 32]", arg24_1: "f32[128, 32, 49, 1]", arg25_1: "f32[128, 32, 49, 1]", arg26_1: "bf16[4096, 32, 49]", arg27_1: "bf16[4096, 49, 32]", arg28_1: "bf16[3072, 1024]", arg29_1: "bf16[6272, 1024]", arg30_1: "f32[1024]", arg31_1: "bf16[128, 7, 7, 1024]", arg32_1: "f32[128, 7, 7, 1]", arg33_1: "f32[128, 7, 7, 1]", arg34_1: "b8[128, 1, 1]", arg35_1: "bf16[1024, 4096]", arg36_1: "bf16[6272, 4096]", arg37_1: "bf16[6272, 4096]", arg38_1: "bf16[4096, 1024]", arg39_1: "bf16[6272, 1024]", arg40_1: "f32[1024]", arg41_1: "bf16[128, 49, 1024]", arg42_1: "f32[128, 49, 1]", arg43_1: "f32[128, 49, 1]", arg44_1: "b8[128, 1, 1, 1]", arg45_1: "bf16[1024, 1024]", arg46_1: "bf16[6272, 1024]", arg47_1: "bf16[4096, 49, 49]", arg48_1: "bf16[4096, 32, 49]", arg49_1: "bf16[4096, 49, 49]", arg50_1: "i64[49, 49]", arg51_1: "f32[169, 32]", arg52_1: "f32[128, 32, 49, 1]", arg53_1: "f32[128, 32, 49, 1]", arg54_1: "bf16[4096, 32, 49]", arg55_1: "bf16[4096, 49, 32]", arg56_1: "bf16[3072, 1024]", arg57_1: "bf16[6272, 1024]", arg58_1: "f32[1024]", arg59_1: "bf16[6272, 1024]", arg60_1: "f32[128, 7, 7, 1]", arg61_1: "f32[128, 7, 7, 1]", arg62_1: "bf16[6272, 2048]", arg63_1: "bf16[1024, 2048]", arg64_1: "f32[2048]", arg65_1: "bf16[128, 7, 7, 2048]", arg66_1: "f32[128, 7, 7, 1]", arg67_1: "f32[128, 7, 7, 1]", arg68_1: "b8[128, 1, 1]", arg69_1: "bf16[512, 2048]", arg70_1: "bf16[25088, 2048]", arg71_1: "bf16[25088, 2048]", arg72_1: "bf16[2048, 512]", arg73_1: "bf16[25088, 512]", arg74_1: "f32[512]", arg75_1: "bf16[128, 196, 512]", arg76_1: "f32[128, 196, 1]", arg77_1: "f32[128, 196, 1]", arg78_1: "b8[128, 1, 1, 1]", arg79_1: "bf16[512, 512]", arg80_1: "bf16[25088, 512]", arg81_1: "bf16[8192, 49, 49]", arg82_1: "bf16[8192, 32, 49]", arg83_1: "bf16[8192, 56, 56]", arg84_1: "i64[49, 49]", arg85_1: "f32[169, 16]", arg86_1: "f32[4, 49, 49]", arg87_1: "f32[512, 16, 49, 1]", arg88_1: "f32[512, 16, 49, 1]", arg89_1: "bf16[8192, 32, 49]", arg90_1: "bf16[8192, 49, 32]", arg91_1: "bf16[1536, 512]", arg92_1: "bf16[25088, 512]", arg93_1: "i64[14]", arg94_1: "f32[512]", arg95_1: "bf16[128, 14, 14, 512]", arg96_1: "f32[128, 14, 14, 1]", arg97_1: "f32[128, 14, 14, 1]", arg98_1: "b8[128, 1, 1]", arg99_1: "bf16[512, 2048]", arg100_1: "bf16[25088, 2048]", arg101_1: "bf16[25088, 2048]", arg102_1: "bf16[2048, 512]", arg103_1: "bf16[25088, 512]", arg104_1: "f32[512]", arg105_1: "bf16[128, 196, 512]", arg106_1: "f32[128, 196, 1]", arg107_1: "f32[128, 196, 1]", arg108_1: "b8[128, 1, 1, 1]", arg109_1: "bf16[512, 512]", arg110_1: "bf16[25088, 512]", arg111_1: "bf16[8192, 49, 49]", arg112_1: "bf16[8192, 32, 49]", arg113_1: "bf16[8192, 56, 56]", arg114_1: "i64[49, 49]", arg115_1: "f32[169, 16]", arg116_1: "f32[512, 16, 49, 1]", arg117_1: "f32[512, 16, 49, 1]", arg118_1: "bf16[8192, 32, 49]", arg119_1: "bf16[8192, 49, 32]", arg120_1: "bf16[1536, 512]", arg121_1: "bf16[25088, 512]", arg122_1: "f32[512]", arg123_1: "bf16[128, 14, 14, 512]", arg124_1: "f32[128, 14, 14, 1]", arg125_1: "f32[128, 14, 14, 1]", arg126_1: "b8[128, 1, 1]", arg127_1: "bf16[512, 2048]", arg128_1: "bf16[25088, 2048]", arg129_1: "bf16[25088, 2048]", arg130_1: "bf16[2048, 512]", arg131_1: "bf16[25088, 512]", arg132_1: "f32[512]", arg133_1: "bf16[128, 196, 512]", arg134_1: "f32[128, 196, 1]", arg135_1: "f32[128, 196, 1]", arg136_1: "b8[128, 1, 1, 1]", arg137_1: "bf16[512, 512]", arg138_1: "bf16[25088, 512]", arg139_1: "bf16[8192, 49, 49]", arg140_1: "bf16[8192, 32, 49]", arg141_1: "bf16[8192, 56, 56]", arg142_1: "i64[49, 49]", arg143_1: "f32[169, 16]", arg144_1: "f32[4, 49, 49]", arg145_1: "f32[512, 16, 49, 1]", arg146_1: "f32[512, 16, 49, 1]", arg147_1: "bf16[8192, 32, 49]", arg148_1: "bf16[8192, 49, 32]", arg149_1: "bf16[1536, 512]", arg150_1: "bf16[25088, 512]", arg151_1: "f32[512]", arg152_1: "bf16[128, 14, 14, 512]", arg153_1: "f32[128, 14, 14, 1]", arg154_1: "f32[128, 14, 14, 1]", arg155_1: "b8[128, 1, 1]", arg156_1: "bf16[512, 2048]", arg157_1: "bf16[25088, 2048]", arg158_1: "bf16[25088, 2048]", arg159_1: "bf16[2048, 512]", arg160_1: "bf16[25088, 512]", arg161_1: "f32[512]", arg162_1: "bf16[128, 196, 512]", arg163_1: "f32[128, 196, 1]", arg164_1: "f32[128, 196, 1]", arg165_1: "b8[128, 1, 1, 1]", arg166_1: "bf16[512, 512]", arg167_1: "bf16[25088, 512]", arg168_1: "bf16[8192, 49, 49]", arg169_1: "bf16[8192, 32, 49]", arg170_1: "bf16[8192, 56, 56]", arg171_1: "i64[49, 49]", arg172_1: "f32[169, 16]", arg173_1: "f32[512, 16, 49, 1]", arg174_1: "f32[512, 16, 49, 1]", arg175_1: "bf16[8192, 32, 49]", arg176_1: "bf16[8192, 49, 32]", arg177_1: "bf16[1536, 512]", arg178_1: "bf16[25088, 512]", arg179_1: "f32[512]", arg180_1: "bf16[128, 14, 14, 512]", arg181_1: "f32[128, 14, 14, 1]", arg182_1: "f32[128, 14, 14, 1]", arg183_1: "b8[128, 1, 1]", arg184_1: "bf16[512, 2048]", arg185_1: "bf16[25088, 2048]", arg186_1: "bf16[25088, 2048]", arg187_1: "bf16[2048, 512]", arg188_1: "bf16[25088, 512]", arg189_1: "f32[512]", arg190_1: "bf16[128, 196, 512]", arg191_1: "f32[128, 196, 1]", arg192_1: "f32[128, 196, 1]", arg193_1: "b8[128, 1, 1, 1]", arg194_1: "bf16[512, 512]", arg195_1: "bf16[25088, 512]", arg196_1: "bf16[8192, 49, 49]", arg197_1: "bf16[8192, 32, 49]", arg198_1: "bf16[8192, 56, 56]", arg199_1: "i64[49, 49]", arg200_1: "f32[169, 16]", arg201_1: "f32[4, 49, 49]", arg202_1: "f32[512, 16, 49, 1]", arg203_1: "f32[512, 16, 49, 1]", arg204_1: "bf16[8192, 32, 49]", arg205_1: "bf16[8192, 49, 32]", arg206_1: "bf16[1536, 512]", arg207_1: "bf16[25088, 512]", arg208_1: "f32[512]", arg209_1: "bf16[128, 14, 14, 512]", arg210_1: "f32[128, 14, 14, 1]", arg211_1: "f32[128, 14, 14, 1]", arg212_1: "b8[128, 1, 1]", arg213_1: "bf16[512, 2048]", arg214_1: "bf16[25088, 2048]", arg215_1: "bf16[25088, 2048]", arg216_1: "bf16[2048, 512]", arg217_1: "bf16[25088, 512]", arg218_1: "f32[512]", arg219_1: "bf16[128, 196, 512]", arg220_1: "f32[128, 196, 1]", arg221_1: "f32[128, 196, 1]", arg222_1: "b8[128, 1, 1, 1]", arg223_1: "bf16[512, 512]", arg224_1: "bf16[25088, 512]", arg225_1: "bf16[8192, 49, 49]", arg226_1: "bf16[8192, 32, 49]", arg227_1: "bf16[8192, 56, 56]", arg228_1: "i64[49, 49]", arg229_1: "f32[169, 16]", arg230_1: "f32[512, 16, 49, 1]", arg231_1: "f32[512, 16, 49, 1]", arg232_1: "bf16[8192, 32, 49]", arg233_1: "bf16[8192, 49, 32]", arg234_1: "bf16[1536, 512]", arg235_1: "bf16[25088, 512]", arg236_1: "f32[512]", arg237_1: "bf16[128, 14, 14, 512]", arg238_1: "f32[128, 14, 14, 1]", arg239_1: "f32[128, 14, 14, 1]", arg240_1: "b8[128, 1, 1]", arg241_1: "bf16[512, 2048]", arg242_1: "bf16[25088, 2048]", arg243_1: "bf16[25088, 2048]", arg244_1: "bf16[2048, 512]", arg245_1: "bf16[25088, 512]", arg246_1: "f32[512]", arg247_1: "bf16[128, 196, 512]", arg248_1: "f32[128, 196, 1]", arg249_1: "f32[128, 196, 1]", arg250_1: "b8[128, 1, 1, 1]", arg251_1: "bf16[512, 512]", arg252_1: "bf16[25088, 512]", arg253_1: "bf16[8192, 49, 49]", arg254_1: "bf16[8192, 32, 49]", arg255_1: "bf16[8192, 56, 56]", arg256_1: "i64[49, 49]", arg257_1: "f32[169, 16]", arg258_1: "f32[4, 49, 49]", arg259_1: "f32[512, 16, 49, 1]", arg260_1: "f32[512, 16, 49, 1]", arg261_1: "bf16[8192, 32, 49]", arg262_1: "bf16[8192, 49, 32]", arg263_1: "bf16[1536, 512]", arg264_1: "bf16[25088, 512]", arg265_1: "f32[512]", arg266_1: "bf16[128, 14, 14, 512]", arg267_1: "f32[128, 14, 14, 1]", arg268_1: "f32[128, 14, 14, 1]", arg269_1: "b8[128, 1, 1]", arg270_1: "bf16[512, 2048]", arg271_1: "bf16[25088, 2048]", arg272_1: "bf16[25088, 2048]", arg273_1: "bf16[2048, 512]", arg274_1: "bf16[25088, 512]", arg275_1: "f32[512]", arg276_1: "bf16[128, 196, 512]", arg277_1: "f32[128, 196, 1]", arg278_1: "f32[128, 196, 1]", arg279_1: "b8[128, 1, 1, 1]", arg280_1: "bf16[512, 512]", arg281_1: "bf16[25088, 512]", arg282_1: "bf16[8192, 49, 49]", arg283_1: "bf16[8192, 32, 49]", arg284_1: "bf16[8192, 56, 56]", arg285_1: "i64[49, 49]", arg286_1: "f32[169, 16]", arg287_1: "f32[512, 16, 49, 1]", arg288_1: "f32[512, 16, 49, 1]", arg289_1: "bf16[8192, 32, 49]", arg290_1: "bf16[8192, 49, 32]", arg291_1: "bf16[1536, 512]", arg292_1: "bf16[25088, 512]", arg293_1: "f32[512]", arg294_1: "bf16[128, 14, 14, 512]", arg295_1: "f32[128, 14, 14, 1]", arg296_1: "f32[128, 14, 14, 1]", arg297_1: "b8[128, 1, 1]", arg298_1: "bf16[512, 2048]", arg299_1: "bf16[25088, 2048]", arg300_1: "bf16[25088, 2048]", arg301_1: "bf16[2048, 512]", arg302_1: "bf16[25088, 512]", arg303_1: "f32[512]", arg304_1: "bf16[128, 196, 512]", arg305_1: "f32[128, 196, 1]", arg306_1: "f32[128, 196, 1]", arg307_1: "b8[128, 1, 1, 1]", arg308_1: "bf16[512, 512]", arg309_1: "bf16[25088, 512]", arg310_1: "bf16[8192, 49, 49]", arg311_1: "bf16[8192, 32, 49]", arg312_1: "bf16[8192, 56, 56]", arg313_1: "i64[49, 49]", arg314_1: "f32[169, 16]", arg315_1: "f32[4, 49, 49]", arg316_1: "f32[512, 16, 49, 1]", arg317_1: "f32[512, 16, 49, 1]", arg318_1: "bf16[8192, 32, 49]", arg319_1: "bf16[8192, 49, 32]", arg320_1: "bf16[1536, 512]", arg321_1: "bf16[25088, 512]", arg322_1: "f32[512]", arg323_1: "bf16[128, 14, 14, 512]", arg324_1: "f32[128, 14, 14, 1]", arg325_1: "f32[128, 14, 14, 1]", arg326_1: "b8[128, 1, 1]", arg327_1: "bf16[512, 2048]", arg328_1: "bf16[25088, 2048]", arg329_1: "bf16[25088, 2048]", arg330_1: "bf16[2048, 512]", arg331_1: "bf16[25088, 512]", arg332_1: "f32[512]", arg333_1: "bf16[128, 196, 512]", arg334_1: "f32[128, 196, 1]", arg335_1: "f32[128, 196, 1]", arg336_1: "b8[128, 1, 1, 1]", arg337_1: "bf16[512, 512]", arg338_1: "bf16[25088, 512]", arg339_1: "bf16[8192, 49, 49]", arg340_1: "bf16[8192, 32, 49]", arg341_1: "bf16[8192, 56, 56]", arg342_1: "i64[49, 49]", arg343_1: "f32[169, 16]", arg344_1: "f32[512, 16, 49, 1]", arg345_1: "f32[512, 16, 49, 1]", arg346_1: "bf16[8192, 32, 49]", arg347_1: "bf16[8192, 49, 32]", arg348_1: "bf16[1536, 512]", arg349_1: "bf16[25088, 512]", arg350_1: "f32[512]", arg351_1: "bf16[128, 14, 14, 512]", arg352_1: "f32[128, 14, 14, 1]", arg353_1: "f32[128, 14, 14, 1]", arg354_1: "b8[128, 1, 1]", arg355_1: "bf16[512, 2048]", arg356_1: "bf16[25088, 2048]", arg357_1: "bf16[25088, 2048]", arg358_1: "bf16[2048, 512]", arg359_1: "bf16[25088, 512]", arg360_1: "f32[512]", arg361_1: "bf16[128, 196, 512]", arg362_1: "f32[128, 196, 1]", arg363_1: "f32[128, 196, 1]", arg364_1: "b8[128, 1, 1, 1]", arg365_1: "bf16[512, 512]", arg366_1: "bf16[25088, 512]", arg367_1: "bf16[8192, 49, 49]", arg368_1: "bf16[8192, 32, 49]", arg369_1: "bf16[8192, 56, 56]", arg370_1: "i64[49, 49]", arg371_1: "f32[169, 16]", arg372_1: "f32[4, 49, 49]", arg373_1: "f32[512, 16, 49, 1]", arg374_1: "f32[512, 16, 49, 1]", arg375_1: "bf16[8192, 32, 49]", arg376_1: "bf16[8192, 49, 32]", arg377_1: "bf16[1536, 512]", arg378_1: "bf16[25088, 512]", arg379_1: "f32[512]", arg380_1: "bf16[128, 14, 14, 512]", arg381_1: "f32[128, 14, 14, 1]", arg382_1: "f32[128, 14, 14, 1]", arg383_1: "b8[128, 1, 1]", arg384_1: "bf16[512, 2048]", arg385_1: "bf16[25088, 2048]", arg386_1: "bf16[25088, 2048]", arg387_1: "bf16[2048, 512]", arg388_1: "bf16[25088, 512]", arg389_1: "f32[512]", arg390_1: "bf16[128, 196, 512]", arg391_1: "f32[128, 196, 1]", arg392_1: "f32[128, 196, 1]", arg393_1: "b8[128, 1, 1, 1]", arg394_1: "bf16[512, 512]", arg395_1: "bf16[25088, 512]", arg396_1: "bf16[8192, 49, 49]", arg397_1: "bf16[8192, 32, 49]", arg398_1: "bf16[8192, 56, 56]", arg399_1: "i64[49, 49]", arg400_1: "f32[169, 16]", arg401_1: "f32[512, 16, 49, 1]", arg402_1: "f32[512, 16, 49, 1]", arg403_1: "bf16[8192, 32, 49]", arg404_1: "bf16[8192, 49, 32]", arg405_1: "bf16[1536, 512]", arg406_1: "bf16[25088, 512]", arg407_1: "f32[512]", arg408_1: "bf16[128, 14, 14, 512]", arg409_1: "f32[128, 14, 14, 1]", arg410_1: "f32[128, 14, 14, 1]", arg411_1: "b8[128, 1, 1]", arg412_1: "bf16[512, 2048]", arg413_1: "bf16[25088, 2048]", arg414_1: "bf16[25088, 2048]", arg415_1: "bf16[2048, 512]", arg416_1: "bf16[25088, 512]", arg417_1: "f32[512]", arg418_1: "bf16[128, 196, 512]", arg419_1: "f32[128, 196, 1]", arg420_1: "f32[128, 196, 1]", arg421_1: "b8[128, 1, 1, 1]", arg422_1: "bf16[512, 512]", arg423_1: "bf16[25088, 512]", arg424_1: "bf16[8192, 49, 49]", arg425_1: "bf16[8192, 32, 49]", arg426_1: "bf16[8192, 56, 56]", arg427_1: "i64[49, 49]", arg428_1: "f32[169, 16]", arg429_1: "f32[4, 49, 49]", arg430_1: "f32[512, 16, 49, 1]", arg431_1: "f32[512, 16, 49, 1]", arg432_1: "bf16[8192, 32, 49]", arg433_1: "bf16[8192, 49, 32]", arg434_1: "bf16[1536, 512]", arg435_1: "bf16[25088, 512]", arg436_1: "f32[512]", arg437_1: "bf16[128, 14, 14, 512]", arg438_1: "f32[128, 14, 14, 1]", arg439_1: "f32[128, 14, 14, 1]", arg440_1: "b8[128, 1, 1]", arg441_1: "bf16[512, 2048]", arg442_1: "bf16[25088, 2048]", arg443_1: "bf16[25088, 2048]", arg444_1: "bf16[2048, 512]", arg445_1: "bf16[25088, 512]", arg446_1: "f32[512]", arg447_1: "bf16[128, 196, 512]", arg448_1: "f32[128, 196, 1]", arg449_1: "f32[128, 196, 1]", arg450_1: "b8[128, 1, 1, 1]", arg451_1: "bf16[512, 512]", arg452_1: "bf16[25088, 512]", arg453_1: "bf16[8192, 49, 49]", arg454_1: "bf16[8192, 32, 49]", arg455_1: "bf16[8192, 56, 56]", arg456_1: "i64[49, 49]", arg457_1: "f32[169, 16]", arg458_1: "f32[512, 16, 49, 1]", arg459_1: "f32[512, 16, 49, 1]", arg460_1: "bf16[8192, 32, 49]", arg461_1: "bf16[8192, 49, 32]", arg462_1: "bf16[1536, 512]", arg463_1: "bf16[25088, 512]", arg464_1: "f32[512]", arg465_1: "bf16[128, 14, 14, 512]", arg466_1: "f32[128, 14, 14, 1]", arg467_1: "f32[128, 14, 14, 1]", arg468_1: "b8[128, 1, 1]", arg469_1: "bf16[512, 2048]", arg470_1: "bf16[25088, 2048]", arg471_1: "bf16[25088, 2048]", arg472_1: "bf16[2048, 512]", arg473_1: "bf16[25088, 512]", arg474_1: "f32[512]", arg475_1: "bf16[128, 196, 512]", arg476_1: "f32[128, 196, 1]", arg477_1: "f32[128, 196, 1]", arg478_1: "b8[128, 1, 1, 1]", arg479_1: "bf16[512, 512]", arg480_1: "bf16[25088, 512]", arg481_1: "bf16[8192, 49, 49]", arg482_1: "bf16[8192, 32, 49]", arg483_1: "bf16[8192, 56, 56]", arg484_1: "i64[49, 49]", arg485_1: "f32[169, 16]", arg486_1: "f32[4, 49, 49]", arg487_1: "f32[512, 16, 49, 1]", arg488_1: "f32[512, 16, 49, 1]", arg489_1: "bf16[8192, 32, 49]", arg490_1: "bf16[8192, 49, 32]", arg491_1: "bf16[1536, 512]", arg492_1: "bf16[25088, 512]", arg493_1: "f32[512]", arg494_1: "bf16[128, 14, 14, 512]", arg495_1: "f32[128, 14, 14, 1]", arg496_1: "f32[128, 14, 14, 1]", arg497_1: "b8[128, 1, 1]", arg498_1: "bf16[512, 2048]", arg499_1: "bf16[25088, 2048]", arg500_1: "bf16[25088, 2048]", arg501_1: "bf16[2048, 512]", arg502_1: "bf16[25088, 512]", arg503_1: "f32[512]", arg504_1: "bf16[128, 196, 512]", arg505_1: "f32[128, 196, 1]", arg506_1: "f32[128, 196, 1]", arg507_1: "b8[128, 1, 1, 1]", arg508_1: "bf16[512, 512]", arg509_1: "bf16[25088, 512]", arg510_1: "bf16[8192, 49, 49]", arg511_1: "bf16[8192, 32, 49]", arg512_1: "bf16[8192, 56, 56]", arg513_1: "i64[49, 49]", arg514_1: "f32[169, 16]", arg515_1: "f32[512, 16, 49, 1]", arg516_1: "f32[512, 16, 49, 1]", arg517_1: "bf16[8192, 32, 49]", arg518_1: "bf16[8192, 49, 32]", arg519_1: "bf16[1536, 512]", arg520_1: "bf16[25088, 512]", arg521_1: "f32[512]", arg522_1: "bf16[128, 14, 14, 512]", arg523_1: "f32[128, 14, 14, 1]", arg524_1: "f32[128, 14, 14, 1]", arg525_1: "b8[128, 1, 1]", arg526_1: "bf16[512, 2048]", arg527_1: "bf16[25088, 2048]", arg528_1: "bf16[25088, 2048]", arg529_1: "bf16[2048, 512]", arg530_1: "bf16[25088, 512]", arg531_1: "f32[512]", arg532_1: "bf16[128, 196, 512]", arg533_1: "f32[128, 196, 1]", arg534_1: "f32[128, 196, 1]", arg535_1: "b8[128, 1, 1, 1]", arg536_1: "bf16[512, 512]", arg537_1: "bf16[25088, 512]", arg538_1: "bf16[8192, 49, 49]", arg539_1: "bf16[8192, 32, 49]", arg540_1: "bf16[8192, 56, 56]", arg541_1: "i64[49, 49]", arg542_1: "f32[169, 16]", arg543_1: "f32[4, 49, 49]", arg544_1: "f32[512, 16, 49, 1]", arg545_1: "f32[512, 16, 49, 1]", arg546_1: "bf16[8192, 32, 49]", arg547_1: "bf16[8192, 49, 32]", arg548_1: "bf16[1536, 512]", arg549_1: "bf16[25088, 512]", arg550_1: "f32[512]", arg551_1: "bf16[128, 14, 14, 512]", arg552_1: "f32[128, 14, 14, 1]", arg553_1: "f32[128, 14, 14, 1]", arg554_1: "b8[128, 1, 1]", arg555_1: "bf16[512, 2048]", arg556_1: "bf16[25088, 2048]", arg557_1: "bf16[25088, 2048]", arg558_1: "bf16[2048, 512]", arg559_1: "bf16[25088, 512]", arg560_1: "f32[512]", arg561_1: "bf16[128, 196, 512]", arg562_1: "f32[128, 196, 1]", arg563_1: "f32[128, 196, 1]", arg564_1: "b8[128, 1, 1, 1]", arg565_1: "bf16[512, 512]", arg566_1: "bf16[25088, 512]", arg567_1: "bf16[8192, 49, 49]", arg568_1: "bf16[8192, 32, 49]", arg569_1: "bf16[8192, 56, 56]", arg570_1: "i64[49, 49]", arg571_1: "f32[169, 16]", arg572_1: "f32[512, 16, 49, 1]", arg573_1: "f32[512, 16, 49, 1]", arg574_1: "bf16[8192, 32, 49]", arg575_1: "bf16[8192, 49, 32]", arg576_1: "bf16[1536, 512]", arg577_1: "bf16[25088, 512]", arg578_1: "f32[512]", arg579_1: "bf16[25088, 512]", arg580_1: "f32[128, 14, 14, 1]", arg581_1: "f32[128, 14, 14, 1]", arg582_1: "bf16[25088, 1024]", arg583_1: "bf16[512, 1024]", arg584_1: "f32[1024]", arg585_1: "bf16[128, 14, 14, 1024]", arg586_1: "f32[128, 14, 14, 1]", arg587_1: "f32[128, 14, 14, 1]", arg588_1: "b8[128, 1, 1]", arg589_1: "bf16[256, 1024]", arg590_1: "bf16[100352, 1024]", arg591_1: "bf16[100352, 1024]", arg592_1: "bf16[1024, 256]", arg593_1: "bf16[100352, 256]", arg594_1: "f32[256]", arg595_1: "bf16[128, 784, 256]", arg596_1: "f32[128, 784, 1]", arg597_1: "f32[128, 784, 1]", arg598_1: "b8[128, 1, 1, 1]", arg599_1: "bf16[256, 256]", arg600_1: "bf16[100352, 256]", arg601_1: "bf16[16384, 49, 49]", arg602_1: "bf16[16384, 32, 49]", arg603_1: "bf16[16384, 56, 56]", arg604_1: "i64[49, 49]", arg605_1: "f32[169, 8]", arg606_1: "f32[16, 49, 49]", arg607_1: "f32[2048, 8, 49, 1]", arg608_1: "f32[2048, 8, 49, 1]", arg609_1: "bf16[16384, 32, 49]", arg610_1: "bf16[16384, 49, 32]", arg611_1: "bf16[768, 256]", arg612_1: "bf16[100352, 256]", arg613_1: "i64[28]", arg614_1: "f32[256]", arg615_1: "bf16[128, 28, 28, 256]", arg616_1: "f32[128, 28, 28, 1]", arg617_1: "f32[128, 28, 28, 1]", arg618_1: "b8[128, 1, 1]", arg619_1: "bf16[256, 1024]", arg620_1: "bf16[100352, 1024]", arg621_1: "bf16[100352, 1024]", arg622_1: "bf16[1024, 256]", arg623_1: "bf16[100352, 256]", arg624_1: "f32[256]", arg625_1: "bf16[128, 784, 256]", arg626_1: "f32[128, 784, 1]", arg627_1: "f32[128, 784, 1]", arg628_1: "b8[128, 1, 1, 1]", arg629_1: "bf16[256, 256]", arg630_1: "bf16[100352, 256]", arg631_1: "bf16[16384, 49, 49]", arg632_1: "bf16[16384, 32, 49]", arg633_1: "bf16[16384, 56, 56]", arg634_1: "i64[49, 49]", arg635_1: "f32[169, 8]", arg636_1: "f32[2048, 8, 49, 1]", arg637_1: "f32[2048, 8, 49, 1]", arg638_1: "bf16[16384, 32, 49]", arg639_1: "bf16[16384, 49, 32]", arg640_1: "bf16[768, 256]", arg641_1: "bf16[100352, 256]", arg642_1: "f32[256]", arg643_1: "bf16[100352, 256]", arg644_1: "f32[128, 28, 28, 1]", arg645_1: "f32[128, 28, 28, 1]", arg646_1: "bf16[100352, 512]", arg647_1: "bf16[256, 512]", arg648_1: "f32[512]", arg649_1: "f32[128, 28, 28, 512]", arg650_1: "f32[128, 28, 28, 1]", arg651_1: "b8[128, 1, 1]", arg652_1: "bf16[128, 512]", arg653_1: "bf16[401408, 512]", arg654_1: "bf16[401408, 512]", arg655_1: "bf16[512, 128]", arg656_1: "bf16[401408, 128]", arg657_1: "f32[128]", arg658_1: "f32[128, 3136, 128]", arg659_1: "f32[128, 3136, 1]", arg660_1: "b8[128, 1, 1, 1]", arg661_1: "bf16[128, 128]", arg662_1: "bf16[401408, 128]", arg663_1: "bf16[32768, 49, 49]", arg664_1: "bf16[32768, 32, 49]", arg665_1: "bf16[32768, 56, 56]", arg666_1: "i64[49, 49]", arg667_1: "f32[169, 4]", arg668_1: "f32[64, 49, 49]", arg669_1: "f32[8192, 4, 49, 1]", arg670_1: "f32[8192, 4, 49, 1]", arg671_1: "bf16[32768, 32, 49]", arg672_1: "bf16[32768, 49, 32]", arg673_1: "bf16[384, 128]", arg674_1: "bf16[401408, 128]", arg675_1: "i64[56]", arg676_1: "f32[128]", arg677_1: "f32[128, 56, 56, 128]", arg678_1: "f32[128, 56, 56, 1]", arg679_1: "bf16[128, 512]", arg680_1: "bf16[401408, 512]", arg681_1: "bf16[401408, 512]", arg682_1: "bf16[512, 128]", arg683_1: "bf16[401408, 128]", arg684_1: "f32[128]", arg685_1: "f32[128, 3136, 128]", arg686_1: "f32[128, 3136, 1]", arg687_1: "bf16[128, 128]", arg688_1: "bf16[401408, 128]", arg689_1: "bf16[32768, 49, 49]", arg690_1: "bf16[32768, 32, 49]", arg691_1: "bf16[32768, 56, 56]", arg692_1: "i64[49, 49]", arg693_1: "f32[169, 4]", arg694_1: "f32[8192, 4, 49, 1]", arg695_1: "f32[8192, 4, 49, 1]", arg696_1: "bf16[32768, 32, 49]", arg697_1: "bf16[32768, 49, 32]", arg698_1: "bf16[384, 128]", arg699_1: "bf16[401408, 128]", arg700_1: "f32[128]", arg701_1: "bf16[128, 128, 56, 56]", arg702_1: "f32[128, 56, 56, 1]", arg703_1: "f32[128, 56, 56, 1]", arg704_1: "f32[128]", arg705_1: "f32[128]", arg706_1: "f32[128, 56, 56, 1]", arg707_1: "f32[128, 56, 56, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31, _shape_param_32, _shape_param_33, _shape_param_34, _shape_param_35, _shape_param_36, _shape_param_37, _shape_param_38, _shape_param_39, _shape_param_40, _shape_param_41, _shape_param_42, _shape_param_43, _shape_param_44, _shape_param_45, _shape_param_46, _shape_param_47, _shape_param_48, _shape_param_49, _shape_param_50, _shape_param_51, _shape_param_52, _shape_param_53, _shape_param_54, _shape_param_55, _shape_param_56, _shape_param_57, _shape_param_58, _shape_param_59, _shape_param_60, _shape_param_61, _shape_param_62, _shape_param_63, _shape_param_64, _shape_param_65, _shape_param_66, _shape_param_67, _shape_param_68, _shape_param_69, _shape_param_70, _shape_param_71, _shape_param_72, _shape_param_73, _shape_param_74, _shape_param_75, _shape_param_76, _shape_param_77, _shape_param_78, _shape_param_79, _shape_param_80, _shape_param_81, _shape_param_82, _shape_param_83, _shape_param_84, _shape_param_85, _shape_param_86, _shape_param_87, _shape_param_88, _shape_param_89, _shape_param_90, _shape_param_91, _shape_param_92, _shape_param_93, _shape_param_94, _shape_param_95, _shape_param_96, _shape_param_97, _shape_param_98, _shape_param_99, _shape_param_100, _shape_param_101, _shape_param_102, _shape_param_103, _shape_param_104, _shape_param_105, _shape_param_106, _shape_param_107, _shape_param_108, _shape_param_109, _shape_param_110, _shape_param_111, _shape_param_112, _shape_param_113, _shape_param_114, _shape_param_115, _shape_param_116, _shape_param_117, _shape_param_118, _shape_param_119, _shape_param_120, _shape_param_121, _shape_param_122, _shape_param_123, _shape_param_124, _shape_param_125, _shape_param_126, _shape_param_127, _shape_param_128, _shape_param_129, _shape_param_130, _shape_param_131, _shape_param_132, _shape_param_133, _shape_param_134, _shape_param_135, _shape_param_136, _shape_param_137, _shape_param_138, _shape_param_139, _shape_param_140, _shape_param_141, _shape_param_142, _shape_param_143, _shape_param_144, _shape_param_145, _shape_param_146, _shape_param_147, _shape_param_148, _shape_param_149, _shape_param_150, _shape_param_151, _shape_param_152, _shape_param_153, _shape_param_154, _shape_param_155, _shape_param_156, _shape_param_157, _shape_param_158, _shape_param_159, _shape_param_160, _shape_param_161, _shape_param_162, _shape_param_163, _shape_param_164, _shape_param_165, _shape_param_166, _shape_param_167, _shape_param_168, _shape_param_169, _shape_param_170, _shape_param_171, _shape_param_172, _shape_param_173, _shape_param_174, _shape_param_175, _shape_param_176, _shape_param_177, _shape_param_178, _shape_param_179, _shape_param_180, _shape_param_181, _shape_param_182, _shape_param_183, _shape_param_184, _shape_param_185, _shape_param_186, _shape_param_187, _shape_param_188, _shape_param_189, _shape_param_190, _shape_param_191, _shape_param_192, _shape_param_193, _shape_param_194, _shape_param_195, _shape_param_196, _shape_param_197, _shape_param_198, _shape_param_199, _shape_param_200, _shape_param_201, _shape_param_202, _shape_param_203, _shape_param_204, _shape_param_205, _shape_param_206, _shape_param_207, _shape_param_208, _shape_param_209, _shape_param_210, _shape_param_211, _shape_param_212, _shape_param_213, _shape_param_214, _shape_param_215, _shape_param_216, _shape_param_217, _shape_param_218, _shape_param_219, _shape_param_220, _shape_param_221, _shape_param_222, _shape_param_223, _shape_param_224, _shape_param_225, _shape_param_226, _shape_param_227, _shape_param_228, _shape_param_229, _shape_param_230, _shape_param_231, _shape_param_232, _shape_param_233, _shape_param_234, _shape_param_235, _shape_param_236, _shape_param_237, _shape_param_238, _shape_param_239, _shape_param_240, _shape_param_241, _shape_param_242, _shape_param_243, _shape_param_244, _shape_param_245, _shape_param_246, _shape_param_247, _shape_param_248, _shape_param_249, _shape_param_250, _shape_param_251, _shape_param_252, _shape_param_253, _shape_param_254, _shape_param_255, _shape_param_256, _shape_param_257, _shape_param_258, _shape_param_259, _shape_param_260, _shape_param_261, _shape_param_262, _shape_param_263, _shape_param_264, _shape_param_265, _shape_param_266, _shape_param_267, _shape_param_268, _shape_param_269, _shape_param_270, _shape_param_271, _shape_param_272, _shape_param_273, _shape_param_274, _shape_param_275, _shape_param_276, _shape_param_277, _shape_param_278, _shape_param_279, _shape_param_280, _shape_param_281, _shape_param_282, _shape_param_283, _shape_param_284, _shape_param_285, _shape_param_286, _shape_param_287, _shape_param_288, _shape_param_289, _shape_param_290, _shape_param_291, _shape_param_292, _shape_param_293, _shape_param_294, _shape_param_295, _shape_param_296, _shape_param_297, _shape_param_298, _shape_param_299, _shape_param_300, _shape_param_301, _shape_param_302, _shape_param_303, _shape_param_304, _shape_param_305, _shape_param_306, _shape_param_307, _shape_param_308, _shape_param_309, _shape_param_310, _shape_param_311, _shape_param_312, _shape_param_313, _shape_param_314, _shape_param_315, _shape_param_316, _shape_param_317, _shape_param_318, _shape_param_319, _shape_param_320, _shape_param_321, _shape_param_322, _shape_param_323, _shape_param_324, _shape_param_325, _shape_param_326, _shape_param_327, _shape_param_328, _shape_param_329, _shape_param_330, _shape_param_331, _shape_param_332, _shape_param_333, _shape_param_334, _shape_param_335, _shape_param_336, _shape_param_337, _shape_param_338, _shape_param_339, _shape_param_340, _shape_param_341, _shape_param_342, _shape_param_343, _shape_param_344, _shape_param_345, _shape_param_346, _shape_param_347, _shape_param_348, _shape_param_349, _shape_param_350, _shape_param_351, _shape_param_352, _shape_param_353, _shape_param_354, _shape_param_355, _shape_param_356, _shape_param_357, _shape_param_358, _shape_param_359, _shape_param_360, _shape_param_361, _shape_param_362, _shape_param_363, _shape_param_364, _shape_param_365, _shape_param_366, _shape_param_367, _shape_param_368, _shape_param_369, _shape_param_370, _shape_param_371, _shape_param_372, _shape_param_373, _shape_param_374, _shape_param_375, _shape_param_376, _shape_param_377, _shape_param_378, _shape_param_379, _shape_param_380, _shape_param_381, _shape_param_382, _shape_param_383, _shape_param_384, _shape_param_385, _shape_param_386, _shape_param_387, _shape_param_388, _shape_param_389, _shape_param_390, _shape_param_391, _shape_param_392, _shape_param_393, _shape_param_394, _shape_param_395, _shape_param_396, _shape_param_397, _shape_param_398, _shape_param_399, _shape_param_400, _shape_param_401, _shape_param_402, _shape_param_403, _shape_param_404, _shape_param_405, _shape_param_406, _shape_param_407, _shape_param_408, _shape_param_409, _shape_param_410, _shape_param_411, _shape_param_412, _shape_param_413, _shape_param_414, _shape_param_415, _shape_param_416, _shape_param_417, _shape_param_418, _shape_param_419, _shape_param_420, _shape_param_421, _shape_param_422, _shape_param_423, _shape_param_424, _shape_param_425, _shape_param_426, _shape_param_427, _shape_param_428, _shape_param_429, _shape_param_430, _shape_param_431, _shape_param_432, _shape_param_433, _shape_param_434, _shape_param_435, _shape_param_436, _shape_param_437, _shape_param_438, _shape_param_439, _shape_param_440, _shape_param_441, _shape_param_442, _shape_param_443, _shape_param_444, _shape_param_445, _shape_param_446, _shape_param_447, _shape_param_448, _shape_param_449, _shape_param_450, _shape_param_451, _shape_param_452, _shape_param_453, _shape_param_454, _shape_param_455, _shape_param_456, _shape_param_457, _shape_param_458, _shape_param_459, _shape_param_460, _shape_param_461, _shape_param_462, _shape_param_463, _shape_param_464, _shape_param_465, _shape_param_466, _shape_param_467, _shape_param_468, _shape_param_469, _shape_param_470, _shape_param_471, _shape_param_472, _shape_param_473, _shape_param_474, _shape_param_475, _shape_param_476, _shape_param_477, _shape_param_478, _shape_param_479, _shape_param_480, _shape_param_481, _shape_param_482, _shape_param_483, _shape_param_484, _shape_param_485, _shape_param_486, _shape_param_487, _shape_param_488, _shape_param_489, _shape_param_490, _shape_param_491, _shape_param_492, _shape_param_493, _shape_param_494, _shape_param_495, _shape_param_496, _shape_param_497, _shape_param_498, _shape_param_499, _shape_param_500, _shape_param_501, _shape_param_502, _shape_param_503, _shape_param_504, _shape_param_505, _shape_param_506, _shape_param_507, _shape_param_508, _shape_param_509, _shape_param_510, _shape_param_511, _shape_param_512, _shape_param_513, _shape_param_514, _shape_param_515, _shape_param_516, _shape_param_517, _shape_param_518, _shape_param_519, _shape_param_520, _shape_param_521, _shape_param_522, _shape_param_523, _shape_param_524, _shape_param_525, _shape_param_526, _shape_param_527, _shape_param_528, _shape_param_529, _shape_param_530, _shape_param_531, _shape_param_532, _shape_param_533, _shape_param_534, _shape_param_535, _shape_param_536, _shape_param_537, _shape_param_538, _shape_param_539, _shape_param_540, _shape_param_541, _shape_param_542, _shape_param_543, _shape_param_544, _shape_param_545, _shape_param_546, _shape_param_547, _shape_param_548, _shape_param_549, _shape_param_550, _shape_param_551, _shape_param_552, _shape_param_553, _shape_param_554, _shape_param_555, _shape_param_556, _shape_param_557, _shape_param_558, _shape_param_559, _shape_param_560, _shape_param_561, _shape_param_562, _shape_param_563, _shape_param_564, _shape_param_565, _shape_param_566, _shape_param_567, _shape_param_568, _shape_param_569, _shape_param_570, _shape_param_571, _shape_param_572, _shape_param_573, _shape_param_574, _shape_param_575, _shape_param_576, _shape_param_577, _shape_param_578, _shape_param_579, _shape_param_580, _shape_param_581, _shape_param_582, _shape_param_583, _shape_param_584, _shape_param_585, _shape_param_586, _shape_param_587, _shape_param_588, _shape_param_589, _shape_param_590, _shape_param_591, _shape_param_592, _shape_param_593, _shape_param_594, _shape_param_595, _shape_param_596, _shape_param_597, _shape_param_598, _shape_param_599, _shape_param_600, _shape_param_601, _shape_param_602, _shape_param_603, _shape_param_604, _shape_param_605, _shape_param_606, _shape_param_607, _shape_param_608, _shape_param_609, _shape_param_610, _shape_param_611, _shape_param_612, _shape_param_613, _shape_param_614, _shape_param_615, _shape_param_616, _shape_param_617, _shape_param_618, _shape_param_619, _shape_param_620, _shape_param_621, _shape_param_622, _shape_param_623, _shape_param_624, _shape_param_625, _shape_param_626, _shape_param_627, _shape_param_628, _shape_param_629, _shape_param_630, _shape_param_631, _shape_param_632, _shape_param_633, _shape_param_634, _shape_param_635, _shape_param_636, _shape_param_637, _shape_param_638, _shape_param_639, _shape_param_640, _shape_param_641, _shape_param_642, _shape_param_643, _shape_param_644, _shape_param_645, _shape_param_646, _shape_param_647, _shape_param_648, _shape_param_649, _shape_param_650, _shape_param_651, _shape_param_652, _shape_param_653, _shape_param_654, _shape_param_655, _shape_param_656, _shape_param_657, _shape_param_658, _shape_param_659, _shape_param_660, _shape_param_661, _shape_param_662, _shape_param_663, _shape_param_664, _shape_param_665, _shape_param_666, _shape_param_667, _shape_param_668, _shape_param_669, _shape_param_670, _shape_param_671, _shape_param_672, _shape_param_673, _shape_param_674, _shape_param_675, _shape_param_676, _shape_param_677, _shape_param_678, _shape_param_679, _shape_param_680, _shape_param_681, _shape_param_682, _shape_param_683, _shape_param_684, _shape_param_685, _shape_param_686, _shape_param_687, _shape_param_688, _shape_param_689, _shape_param_690, _shape_param_691, _shape_param_692, _shape_param_693, _shape_param_694, _shape_param_695, _shape_param_696, _shape_param_697, _shape_param_698, _shape_param_699, _shape_param_700, _shape_param_701, _shape_param_702, _shape_param_703, _shape_param_704, _shape_param_705, _shape_param_706, _shape_param_707, _shape_param_708, _shape_param_709, _shape_param_710, _shape_param_711, _shape_param_712, _shape_param_713, _shape_param_714, _shape_param_715, _shape_param_716, _shape_param_717, _shape_param_718, _shape_param_719, _shape_param_720, _shape_param_721, _shape_param_722, _shape_param_723, _shape_param_724, _shape_param_725, _shape_param_726, _shape_param_727, _shape_param_728, _shape_param_729, _shape_param_730, _shape_param_731, _shape_param_732, _shape_param_733, _shape_param_734, _shape_param_735, _shape_param_736, _shape_param_737, _shape_param_738, _shape_param_739, _shape_param_740, _shape_param_741, _shape_param_742, _shape_param_743, _shape_param_744, _shape_param_745, _shape_param_746, _shape_param_747, _shape_param_748, _shape_param_749, _shape_param_750, _shape_param_751, _shape_param_752, _shape_param_753, _shape_param_754, _shape_param_755, _shape_param_756, _shape_param_757, _shape_param_758, _shape_param_759, _shape_param_760, _shape_param_761, _shape_param_762, _shape_param_763, _shape_param_764, _shape_param_765, _shape_param_766, _shape_param_767, _shape_param_768, _shape_param_769, _shape_param_770, _shape_param_771, _shape_param_772, _shape_param_773, _shape_param_774, _shape_param_775, _shape_param_776, _shape_param_777, _shape_param_778, _shape_param_779, _shape_param_780, _shape_param_781, _shape_param_782, _shape_param_783, _shape_param_784, _shape_param_785, _shape_param_786, _shape_param_787, _shape_param_788, _shape_param_789, _shape_param_790, _shape_param_791, _shape_param_792, _shape_param_793, _shape_param_794, _shape_param_795, _shape_param_796, _shape_param_797, _shape_param_798, _shape_param_799, _shape_param_800, _shape_param_801, _shape_param_802, _shape_param_803, _shape_param_804, _shape_param_805, _shape_param_806, _shape_param_807, _shape_param_808, _shape_param_809, _shape_param_810, _shape_param_811, _shape_param_812, _shape_param_813, _shape_param_814, _shape_param_815, _shape_param_816, _shape_param_817, _shape_param_818, _shape_param_819, _shape_param_820, _shape_param_821, _shape_param_822, _shape_param_823, _shape_param_824, _shape_param_825, _shape_param_826, _shape_param_827, _shape_param_828, _shape_param_829, _shape_param_830, _shape_param_831, _shape_param_832, _shape_param_833):
        # No stacktrace found for following nodes
        mm: "bf16[128, 1024]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        convert_element_type: "f32[128, 1024]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        unsqueeze: "f32[128, 1, 1024]" = torch.ops.aten.unsqueeze.default(convert_element_type, 1);  convert_element_type = None
        unsqueeze_1: "f32[128, 1, 1, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        expand: "f32[128, 7, 7, 1024]" = torch.ops.aten.expand.default(unsqueeze_1, _shape_param_0);  unsqueeze_1 = _shape_param_0 = None
        div: "f32[128, 7, 7, 1024]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None
        mul: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(div, arg2_1);  arg2_1 = None
        mul_1: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul, 1024)
        sum_1: "f32[128, 7, 7, 1]" = torch.ops.aten.sum.dim_IntList(mul, [3], True)
        convert_element_type_1: "f32[128, 7, 7, 1024]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.float32);  arg3_1 = None
        sub: "f32[128, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_1, arg4_1);  convert_element_type_1 = arg4_1 = None
        mul_2: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(sub, arg5_1);  sub = None
        mul_3: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul, mul_2);  mul = None
        sum_2: "f32[128, 7, 7, 1]" = torch.ops.aten.sum.dim_IntList(mul_3, [3], True);  mul_3 = None
        mul_4: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_2, sum_2);  sum_2 = None
        sub_1: "f32[128, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(mul_1, sum_1);  mul_1 = sum_1 = None
        sub_2: "f32[128, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(sub_1, mul_4);  sub_1 = mul_4 = None
        div_1: "f32[128, 7, 7, 1]" = torch.ops.aten.div.Tensor(arg5_1, 1024);  arg5_1 = None
        mul_5: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(div_1, sub_2);  div_1 = sub_2 = None
        mul_6: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(div, mul_2);  mul_2 = None
        sum_3: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_6, [0, 1, 2]);  mul_6 = None
        sum_4: "f32[1024]" = torch.ops.aten.sum.dim_IntList(div, [0, 1, 2]);  div = None
        convert_element_type_2: "bf16[128, 7, 7, 1024]" = torch.ops.prims.convert_element_type.default(mul_5, torch.bfloat16);  mul_5 = None
        view: "bf16[128, 49, 1024]" = torch.ops.aten.view.default(convert_element_type_2, _shape_param_1);  convert_element_type_2 = _shape_param_1 = None
        convert_element_type_3: "bf16[128, 1, 1]" = torch.ops.prims.convert_element_type.default(arg6_1, torch.bfloat16);  arg6_1 = None
        div_2: "bf16[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_3, 0.8999999985098839);  convert_element_type_3 = None
        mul_7: "bf16[128, 49, 1024]" = torch.ops.aten.mul.Tensor(view, div_2);  div_2 = None
        view_1: "bf16[6272, 1024]" = torch.ops.aten.view.default(mul_7, _shape_param_2);  mul_7 = _shape_param_2 = None
        mm_1: "bf16[6272, 4096]" = torch.ops.aten.mm.default(view_1, arg7_1);  arg7_1 = None
        permute: "bf16[1024, 6272]" = torch.ops.aten.permute.default(view_1, [1, 0])
        mm_2: "bf16[1024, 4096]" = torch.ops.aten.mm.default(permute, arg8_1);  permute = arg8_1 = None
        sum_5: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1, [0], True, dtype = torch.float32);  view_1 = None
        view_2: "f32[1024]" = torch.ops.aten.view.default(sum_5, _shape_param_3);  sum_5 = _shape_param_3 = None
        convert_element_type_4: "bf16[1024]" = torch.ops.prims.convert_element_type.default(view_2, torch.bfloat16);  view_2 = None
        view_3: "bf16[128, 49, 4096]" = torch.ops.aten.view.default(mm_1, _shape_param_4);  mm_1 = _shape_param_4 = None
        convert_element_type_5: "f32[1024, 4096]" = torch.ops.prims.convert_element_type.default(mm_2, torch.float32);  mm_2 = None
        convert_element_type_6: "f32[1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_4, torch.float32);  convert_element_type_4 = None
        convert_element_type_7: "f32[128, 49, 4096]" = torch.ops.prims.convert_element_type.default(view_3, torch.float32);  view_3 = None
        view_4: "bf16[128, 49, 4096]" = torch.ops.aten.view.default(arg9_1, _shape_param_5);  arg9_1 = _shape_param_5 = None
        convert_element_type_8: "f32[128, 49, 4096]" = torch.ops.prims.convert_element_type.default(view_4, torch.float32);  view_4 = None
        mul_8: "f32[128, 49, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_8, 0.7071067811865476)
        erf: "f32[128, 49, 4096]" = torch.ops.aten.erf.default(mul_8);  mul_8 = None
        add: "f32[128, 49, 4096]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_9: "f32[128, 49, 4096]" = torch.ops.aten.mul.Tensor(add, 0.5);  add = None
        mul_10: "f32[128, 49, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_8, convert_element_type_8)
        mul_11: "f32[128, 49, 4096]" = torch.ops.aten.mul.Tensor(mul_10, -0.5);  mul_10 = None
        exp: "f32[128, 49, 4096]" = torch.ops.aten.exp.default(mul_11);  mul_11 = None
        mul_12: "f32[128, 49, 4096]" = torch.ops.aten.mul.Tensor(exp, 0.3989422804014327);  exp = None
        mul_13: "f32[128, 49, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_8, mul_12);  convert_element_type_8 = mul_12 = None
        add_1: "f32[128, 49, 4096]" = torch.ops.aten.add.Tensor(mul_9, mul_13);  mul_9 = mul_13 = None
        mul_14: "f32[128, 49, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_7, add_1);  convert_element_type_7 = add_1 = None
        convert_element_type_9: "bf16[128, 49, 4096]" = torch.ops.prims.convert_element_type.default(mul_14, torch.bfloat16);  mul_14 = None
        view_5: "bf16[6272, 4096]" = torch.ops.aten.view.default(convert_element_type_9, _shape_param_6);  convert_element_type_9 = _shape_param_6 = None
        mm_3: "bf16[6272, 1024]" = torch.ops.aten.mm.default(view_5, arg10_1);  arg10_1 = None
        permute_1: "bf16[4096, 6272]" = torch.ops.aten.permute.default(view_5, [1, 0])
        mm_4: "bf16[4096, 1024]" = torch.ops.aten.mm.default(permute_1, arg11_1);  permute_1 = arg11_1 = None
        sum_6: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_5, [0], True, dtype = torch.float32);  view_5 = None
        view_6: "f32[4096]" = torch.ops.aten.view.default(sum_6, _shape_param_7);  sum_6 = _shape_param_7 = None
        convert_element_type_10: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_6, torch.bfloat16);  view_6 = None
        view_7: "bf16[128, 49, 1024]" = torch.ops.aten.view.default(mm_3, _shape_param_8);  mm_3 = _shape_param_8 = None
        convert_element_type_11: "f32[128, 49, 1024]" = torch.ops.prims.convert_element_type.default(view_7, torch.float32);  view_7 = None
        convert_element_type_12: "f32[4096, 1024]" = torch.ops.prims.convert_element_type.default(mm_4, torch.float32);  mm_4 = None
        convert_element_type_13: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_10, torch.float32);  convert_element_type_10 = None
        mul_15: "f32[128, 49, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_11, arg12_1);  arg12_1 = None
        mul_16: "f32[128, 49, 1024]" = torch.ops.aten.mul.Tensor(mul_15, 1024)
        sum_7: "f32[128, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_15, [2], True)
        convert_element_type_14: "f32[128, 49, 1024]" = torch.ops.prims.convert_element_type.default(arg13_1, torch.float32);  arg13_1 = None
        sub_3: "f32[128, 49, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_14, arg14_1);  convert_element_type_14 = arg14_1 = None
        mul_17: "f32[128, 49, 1024]" = torch.ops.aten.mul.Tensor(sub_3, arg15_1);  sub_3 = None
        mul_18: "f32[128, 49, 1024]" = torch.ops.aten.mul.Tensor(mul_15, mul_17);  mul_15 = None
        sum_8: "f32[128, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_18, [2], True);  mul_18 = None
        mul_19: "f32[128, 49, 1024]" = torch.ops.aten.mul.Tensor(mul_17, sum_8);  sum_8 = None
        sub_4: "f32[128, 49, 1024]" = torch.ops.aten.sub.Tensor(mul_16, sum_7);  mul_16 = sum_7 = None
        sub_5: "f32[128, 49, 1024]" = torch.ops.aten.sub.Tensor(sub_4, mul_19);  sub_4 = mul_19 = None
        div_3: "f32[128, 49, 1]" = torch.ops.aten.div.Tensor(arg15_1, 1024);  arg15_1 = None
        mul_20: "f32[128, 49, 1024]" = torch.ops.aten.mul.Tensor(div_3, sub_5);  div_3 = sub_5 = None
        mul_21: "f32[128, 49, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_11, mul_17);  mul_17 = None
        sum_9: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_21, [0, 1]);  mul_21 = None
        sum_10: "f32[1024]" = torch.ops.aten.sum.dim_IntList(convert_element_type_11, [0, 1]);  convert_element_type_11 = None
        convert_element_type_15: "bf16[128, 49, 1024]" = torch.ops.prims.convert_element_type.default(mul_20, torch.bfloat16);  mul_20 = None
        add_2: "bf16[128, 49, 1024]" = torch.ops.aten.add.Tensor(view, convert_element_type_15);  view = convert_element_type_15 = None
        view_8: "bf16[128, 7, 7, 1024]" = torch.ops.aten.view.default(add_2, _shape_param_9);  add_2 = _shape_param_9 = None
        convert_element_type_16: "bf16[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(arg16_1, torch.bfloat16);  arg16_1 = None
        div_4: "bf16[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_16, 0.8999999985098839);  convert_element_type_16 = None
        mul_22: "bf16[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(view_8, div_4);  div_4 = None
        view_9: "bf16[128, 1, 7, 1, 7, 1024]" = torch.ops.aten.view.default(mul_22, _shape_param_10);  mul_22 = _shape_param_10 = None
        permute_2: "bf16[128, 1, 1, 7, 7, 1024]" = torch.ops.aten.permute.default(view_9, [0, 1, 3, 2, 4, 5]);  view_9 = None
        view_10: "bf16[128, 7, 7, 1024]" = torch.ops.aten.view.default(permute_2, _shape_param_11);  permute_2 = _shape_param_11 = None
        view_11: "bf16[128, 49, 1024]" = torch.ops.aten.view.default(view_10, _shape_param_12);  view_10 = _shape_param_12 = None
        view_12: "bf16[6272, 1024]" = torch.ops.aten.view.default(view_11, _shape_param_13);  view_11 = _shape_param_13 = None
        mm_5: "bf16[6272, 1024]" = torch.ops.aten.mm.default(view_12, arg17_1);  arg17_1 = None
        permute_3: "bf16[1024, 6272]" = torch.ops.aten.permute.default(view_12, [1, 0])
        mm_6: "bf16[1024, 1024]" = torch.ops.aten.mm.default(permute_3, arg18_1);  permute_3 = arg18_1 = None
        sum_11: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_12, [0], True, dtype = torch.float32);  view_12 = None
        view_13: "f32[1024]" = torch.ops.aten.view.default(sum_11, _shape_param_14);  sum_11 = _shape_param_14 = None
        convert_element_type_17: "bf16[1024]" = torch.ops.prims.convert_element_type.default(view_13, torch.bfloat16);  view_13 = None
        view_14: "bf16[128, 49, 1024]" = torch.ops.aten.view.default(mm_5, _shape_param_15);  mm_5 = _shape_param_15 = None
        convert_element_type_18: "f32[1024, 1024]" = torch.ops.prims.convert_element_type.default(mm_6, torch.float32);  mm_6 = None
        convert_element_type_19: "f32[1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_17, torch.float32);  convert_element_type_17 = None
        view_15: "bf16[128, 49, 32, 32]" = torch.ops.aten.view.default(view_14, _shape_param_16);  view_14 = _shape_param_16 = None
        permute_4: "bf16[128, 32, 49, 32]" = torch.ops.aten.permute.default(view_15, [0, 2, 1, 3]);  view_15 = None
        clone: "bf16[128, 32, 49, 32]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
        view_16: "bf16[4096, 49, 32]" = torch.ops.aten.view.default(clone, _shape_param_17);  clone = _shape_param_17 = None
        bmm: "bf16[4096, 49, 32]" = torch.ops.aten.bmm.default(arg19_1, view_16);  arg19_1 = None
        bmm_1: "bf16[4096, 49, 49]" = torch.ops.aten.bmm.default(view_16, arg20_1);  view_16 = arg20_1 = None
        view_17: "bf16[128, 32, 49, 32]" = torch.ops.aten.view.default(bmm, _shape_param_18);  bmm = _shape_param_18 = None
        view_18: "bf16[128, 32, 49, 49]" = torch.ops.aten.view.default(bmm_1, _shape_param_19);  bmm_1 = _shape_param_19 = None
        convert_element_type_20: "f32[128, 32, 49, 49]" = torch.ops.prims.convert_element_type.default(view_18, torch.float32);  view_18 = None
        view_19: "bf16[128, 32, 49, 49]" = torch.ops.aten.view.default(arg21_1, _shape_param_20);  arg21_1 = _shape_param_20 = None
        view_20: "i64[2401]" = torch.ops.aten.view.default(arg22_1, [-1]);  arg22_1 = None
        index: "f32[2401, 32]" = torch.ops.aten.index.Tensor(arg23_1, [view_20]);  arg23_1 = None
        view_21: "f32[49, 49, 32]" = torch.ops.aten.view.default(index, _shape_param_21);  index = _shape_param_21 = None
        permute_5: "f32[32, 49, 49]" = torch.ops.aten.permute.default(view_21, [2, 0, 1]);  view_21 = None
        clone_1: "f32[32, 49, 49]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
        unsqueeze_2: "f32[1, 32, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_1, 0);  clone_1 = None
        add_3: "f32[128, 32, 49, 49]" = torch.ops.aten.add.Tensor(view_19, unsqueeze_2);  view_19 = unsqueeze_2 = None
        sub_6: "f32[128, 32, 49, 49]" = torch.ops.aten.sub.Tensor(add_3, arg24_1);  add_3 = arg24_1 = None
        exp_1: "f32[128, 32, 49, 49]" = torch.ops.aten.exp.default(sub_6);  sub_6 = None
        div_5: "f32[128, 32, 49, 49]" = torch.ops.aten.div.Tensor(exp_1, arg25_1);  exp_1 = arg25_1 = None
        mul_23: "f32[128, 32, 49, 49]" = torch.ops.aten.mul.Tensor(convert_element_type_20, div_5);  convert_element_type_20 = None
        sum_12: "f32[128, 32, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_23, [-1], True)
        neg: "f32[128, 32, 49, 49]" = torch.ops.aten.neg.default(div_5);  div_5 = None
        fma: "f32[128, 32, 49, 49]" = torch.ops.prims.fma.default(neg, sum_12, mul_23);  neg = sum_12 = mul_23 = None
        convert_element_type_21: "bf16[128, 32, 49, 49]" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16)
        sum_13: "f32[1, 32, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma, [0], True, dtype = torch.float32);  fma = None
        squeeze: "f32[32, 49, 49]" = torch.ops.aten.squeeze.dim(sum_13, 0);  sum_13 = None
        permute_6: "f32[49, 49, 32]" = torch.ops.aten.permute.default(squeeze, [1, 2, 0]);  squeeze = None
        view_22: "f32[2401, 32]" = torch.ops.aten.view.default(permute_6, _shape_param_22);  permute_6 = _shape_param_22 = None
        full: "f32[169, 32]" = torch.ops.aten.full.default(_shape_param_23, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_23 = None
        index_put: "f32[169, 32]" = torch.ops.aten.index_put.default(full, [view_20], view_22, True);  view_20 = view_22 = None
        view_23: "bf16[4096, 49, 49]" = torch.ops.aten.view.default(convert_element_type_21, _shape_param_24);  convert_element_type_21 = _shape_param_24 = None
        bmm_2: "bf16[4096, 32, 49]" = torch.ops.aten.bmm.default(arg26_1, view_23);  arg26_1 = None
        bmm_3: "bf16[4096, 49, 32]" = torch.ops.aten.bmm.default(view_23, arg27_1);  view_23 = arg27_1 = None
        view_24: "bf16[128, 32, 32, 49]" = torch.ops.aten.view.default(bmm_2, _shape_param_25);  bmm_2 = _shape_param_25 = None
        view_25: "bf16[128, 32, 49, 32]" = torch.ops.aten.view.default(bmm_3, _shape_param_26);  bmm_3 = _shape_param_26 = None
        permute_7: "bf16[128, 32, 49, 32]" = torch.ops.aten.permute.default(view_24, [0, 1, 3, 2]);  view_24 = None
        mul_24: "bf16[128, 32, 49, 32]" = torch.ops.aten.mul.Tensor(view_25, 0.1767766952966369);  view_25 = None
        cat: "bf16[384, 32, 49, 32]" = torch.ops.aten.cat.default([mul_24, permute_7, view_17]);  mul_24 = permute_7 = view_17 = None
        view_26: "bf16[3, 128, 32, 49, 32]" = torch.ops.aten.view.default(cat, _shape_param_27);  cat = _shape_param_27 = None
        permute_8: "bf16[128, 49, 3, 32, 32]" = torch.ops.aten.permute.default(view_26, [1, 3, 0, 2, 4]);  view_26 = None
        clone_2: "bf16[128, 49, 3, 32, 32]" = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None
        view_27: "bf16[128, 49, 3072]" = torch.ops.aten.view.default(clone_2, _shape_param_28);  clone_2 = _shape_param_28 = None
        view_28: "bf16[6272, 3072]" = torch.ops.aten.view.default(view_27, _shape_param_29);  view_27 = _shape_param_29 = None
        mm_7: "bf16[6272, 1024]" = torch.ops.aten.mm.default(view_28, arg28_1);  arg28_1 = None
        permute_9: "bf16[3072, 6272]" = torch.ops.aten.permute.default(view_28, [1, 0])
        mm_8: "bf16[3072, 1024]" = torch.ops.aten.mm.default(permute_9, arg29_1);  permute_9 = arg29_1 = None
        sum_14: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_28, [0], True, dtype = torch.float32);  view_28 = None
        view_29: "f32[3072]" = torch.ops.aten.view.default(sum_14, _shape_param_30);  sum_14 = _shape_param_30 = None
        convert_element_type_22: "bf16[3072]" = torch.ops.prims.convert_element_type.default(view_29, torch.bfloat16);  view_29 = None
        view_30: "bf16[128, 49, 1024]" = torch.ops.aten.view.default(mm_7, _shape_param_31);  mm_7 = _shape_param_31 = None
        convert_element_type_23: "f32[128, 49, 1024]" = torch.ops.prims.convert_element_type.default(view_30, torch.float32);  view_30 = None
        convert_element_type_24: "f32[3072, 1024]" = torch.ops.prims.convert_element_type.default(mm_8, torch.float32);  mm_8 = None
        convert_element_type_25: "f32[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_22, torch.float32);  convert_element_type_22 = None
        view_31: "f32[128, 7, 7, 1024]" = torch.ops.aten.view.default(convert_element_type_23, _shape_param_32);  convert_element_type_23 = _shape_param_32 = None
        view_32: "f32[128, 1, 1, 7, 7, 1024]" = torch.ops.aten.view.default(view_31, _shape_param_33);  view_31 = _shape_param_33 = None
        permute_10: "f32[128, 1, 7, 1, 7, 1024]" = torch.ops.aten.permute.default(view_32, [0, 1, 3, 2, 4, 5]);  view_32 = None
        view_33: "f32[128, 7, 7, 1024]" = torch.ops.aten.view.default(permute_10, _shape_param_34);  permute_10 = _shape_param_34 = None
        mul_25: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(view_33, arg30_1);  arg30_1 = None
        mul_26: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_25, 1024)
        sum_15: "f32[128, 7, 7, 1]" = torch.ops.aten.sum.dim_IntList(mul_25, [3], True)
        convert_element_type_26: "f32[128, 7, 7, 1024]" = torch.ops.prims.convert_element_type.default(arg31_1, torch.float32);  arg31_1 = None
        sub_7: "f32[128, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_26, arg32_1);  convert_element_type_26 = arg32_1 = None
        mul_27: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(sub_7, arg33_1);  sub_7 = None
        mul_28: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_25, mul_27);  mul_25 = None
        sum_16: "f32[128, 7, 7, 1]" = torch.ops.aten.sum.dim_IntList(mul_28, [3], True);  mul_28 = None
        mul_29: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_27, sum_16);  sum_16 = None
        sub_8: "f32[128, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(mul_26, sum_15);  mul_26 = sum_15 = None
        sub_9: "f32[128, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(sub_8, mul_29);  sub_8 = mul_29 = None
        div_6: "f32[128, 7, 7, 1]" = torch.ops.aten.div.Tensor(arg33_1, 1024);  arg33_1 = None
        mul_30: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(div_6, sub_9);  div_6 = sub_9 = None
        mul_31: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(view_33, mul_27);  mul_27 = None
        sum_17: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_31, [0, 1, 2]);  mul_31 = None
        sum_18: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_33, [0, 1, 2]);  view_33 = None
        convert_element_type_27: "bf16[128, 7, 7, 1024]" = torch.ops.prims.convert_element_type.default(mul_30, torch.bfloat16);  mul_30 = None
        add_4: "bf16[128, 7, 7, 1024]" = torch.ops.aten.add.Tensor(view_8, convert_element_type_27);  view_8 = convert_element_type_27 = None
        view_34: "bf16[128, 49, 1024]" = torch.ops.aten.view.default(add_4, _shape_param_35);  add_4 = _shape_param_35 = None
        convert_element_type_28: "bf16[128, 1, 1]" = torch.ops.prims.convert_element_type.default(arg34_1, torch.bfloat16);  arg34_1 = None
        div_7: "bf16[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_28, 0.9043478220701218);  convert_element_type_28 = None
        mul_32: "bf16[128, 49, 1024]" = torch.ops.aten.mul.Tensor(view_34, div_7);  div_7 = None
        view_35: "bf16[6272, 1024]" = torch.ops.aten.view.default(mul_32, _shape_param_36);  mul_32 = _shape_param_36 = None
        mm_9: "bf16[6272, 4096]" = torch.ops.aten.mm.default(view_35, arg35_1);  arg35_1 = None
        permute_11: "bf16[1024, 6272]" = torch.ops.aten.permute.default(view_35, [1, 0])
        mm_10: "bf16[1024, 4096]" = torch.ops.aten.mm.default(permute_11, arg36_1);  permute_11 = arg36_1 = None
        sum_19: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_35, [0], True, dtype = torch.float32);  view_35 = None
        view_36: "f32[1024]" = torch.ops.aten.view.default(sum_19, _shape_param_37);  sum_19 = _shape_param_37 = None
        convert_element_type_29: "bf16[1024]" = torch.ops.prims.convert_element_type.default(view_36, torch.bfloat16);  view_36 = None
        view_37: "bf16[128, 49, 4096]" = torch.ops.aten.view.default(mm_9, _shape_param_38);  mm_9 = _shape_param_38 = None
        convert_element_type_30: "f32[1024, 4096]" = torch.ops.prims.convert_element_type.default(mm_10, torch.float32);  mm_10 = None
        convert_element_type_31: "f32[1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_29, torch.float32);  convert_element_type_29 = None
        convert_element_type_32: "f32[128, 49, 4096]" = torch.ops.prims.convert_element_type.default(view_37, torch.float32);  view_37 = None
        view_38: "bf16[128, 49, 4096]" = torch.ops.aten.view.default(arg37_1, _shape_param_39);  arg37_1 = _shape_param_39 = None
        convert_element_type_33: "f32[128, 49, 4096]" = torch.ops.prims.convert_element_type.default(view_38, torch.float32);  view_38 = None
        mul_33: "f32[128, 49, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_33, 0.7071067811865476)
        erf_1: "f32[128, 49, 4096]" = torch.ops.aten.erf.default(mul_33);  mul_33 = None
        add_5: "f32[128, 49, 4096]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_34: "f32[128, 49, 4096]" = torch.ops.aten.mul.Tensor(add_5, 0.5);  add_5 = None
        mul_35: "f32[128, 49, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_33, convert_element_type_33)
        mul_36: "f32[128, 49, 4096]" = torch.ops.aten.mul.Tensor(mul_35, -0.5);  mul_35 = None
        exp_2: "f32[128, 49, 4096]" = torch.ops.aten.exp.default(mul_36);  mul_36 = None
        mul_37: "f32[128, 49, 4096]" = torch.ops.aten.mul.Tensor(exp_2, 0.3989422804014327);  exp_2 = None
        mul_38: "f32[128, 49, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_33, mul_37);  convert_element_type_33 = mul_37 = None
        add_6: "f32[128, 49, 4096]" = torch.ops.aten.add.Tensor(mul_34, mul_38);  mul_34 = mul_38 = None
        mul_39: "f32[128, 49, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_32, add_6);  convert_element_type_32 = add_6 = None
        convert_element_type_34: "bf16[128, 49, 4096]" = torch.ops.prims.convert_element_type.default(mul_39, torch.bfloat16);  mul_39 = None
        view_39: "bf16[6272, 4096]" = torch.ops.aten.view.default(convert_element_type_34, _shape_param_40);  convert_element_type_34 = _shape_param_40 = None
        mm_11: "bf16[6272, 1024]" = torch.ops.aten.mm.default(view_39, arg38_1);  arg38_1 = None
        permute_12: "bf16[4096, 6272]" = torch.ops.aten.permute.default(view_39, [1, 0])
        mm_12: "bf16[4096, 1024]" = torch.ops.aten.mm.default(permute_12, arg39_1);  permute_12 = arg39_1 = None
        sum_20: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_39, [0], True, dtype = torch.float32);  view_39 = None
        view_40: "f32[4096]" = torch.ops.aten.view.default(sum_20, _shape_param_41);  sum_20 = _shape_param_41 = None
        convert_element_type_35: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_40, torch.bfloat16);  view_40 = None
        view_41: "bf16[128, 49, 1024]" = torch.ops.aten.view.default(mm_11, _shape_param_42);  mm_11 = _shape_param_42 = None
        convert_element_type_36: "f32[128, 49, 1024]" = torch.ops.prims.convert_element_type.default(view_41, torch.float32);  view_41 = None
        convert_element_type_37: "f32[4096, 1024]" = torch.ops.prims.convert_element_type.default(mm_12, torch.float32);  mm_12 = None
        convert_element_type_38: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_35, torch.float32);  convert_element_type_35 = None
        mul_40: "f32[128, 49, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_36, arg40_1);  arg40_1 = None
        mul_41: "f32[128, 49, 1024]" = torch.ops.aten.mul.Tensor(mul_40, 1024)
        sum_21: "f32[128, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_40, [2], True)
        convert_element_type_39: "f32[128, 49, 1024]" = torch.ops.prims.convert_element_type.default(arg41_1, torch.float32);  arg41_1 = None
        sub_10: "f32[128, 49, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_39, arg42_1);  convert_element_type_39 = arg42_1 = None
        mul_42: "f32[128, 49, 1024]" = torch.ops.aten.mul.Tensor(sub_10, arg43_1);  sub_10 = None
        mul_43: "f32[128, 49, 1024]" = torch.ops.aten.mul.Tensor(mul_40, mul_42);  mul_40 = None
        sum_22: "f32[128, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_43, [2], True);  mul_43 = None
        mul_44: "f32[128, 49, 1024]" = torch.ops.aten.mul.Tensor(mul_42, sum_22);  sum_22 = None
        sub_11: "f32[128, 49, 1024]" = torch.ops.aten.sub.Tensor(mul_41, sum_21);  mul_41 = sum_21 = None
        sub_12: "f32[128, 49, 1024]" = torch.ops.aten.sub.Tensor(sub_11, mul_44);  sub_11 = mul_44 = None
        div_8: "f32[128, 49, 1]" = torch.ops.aten.div.Tensor(arg43_1, 1024);  arg43_1 = None
        mul_45: "f32[128, 49, 1024]" = torch.ops.aten.mul.Tensor(div_8, sub_12);  div_8 = sub_12 = None
        mul_46: "f32[128, 49, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_36, mul_42);  mul_42 = None
        sum_23: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_46, [0, 1]);  mul_46 = None
        sum_24: "f32[1024]" = torch.ops.aten.sum.dim_IntList(convert_element_type_36, [0, 1]);  convert_element_type_36 = None
        convert_element_type_40: "bf16[128, 49, 1024]" = torch.ops.prims.convert_element_type.default(mul_45, torch.bfloat16);  mul_45 = None
        add_7: "bf16[128, 49, 1024]" = torch.ops.aten.add.Tensor(view_34, convert_element_type_40);  view_34 = convert_element_type_40 = None
        view_42: "bf16[128, 7, 7, 1024]" = torch.ops.aten.view.default(add_7, _shape_param_43);  add_7 = _shape_param_43 = None
        convert_element_type_41: "bf16[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(arg44_1, torch.bfloat16);  arg44_1 = None
        div_9: "bf16[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_41, 0.9043478220701218);  convert_element_type_41 = None
        mul_47: "bf16[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(view_42, div_9);  div_9 = None
        view_43: "bf16[128, 1, 7, 1, 7, 1024]" = torch.ops.aten.view.default(mul_47, _shape_param_44);  mul_47 = _shape_param_44 = None
        permute_13: "bf16[128, 1, 1, 7, 7, 1024]" = torch.ops.aten.permute.default(view_43, [0, 1, 3, 2, 4, 5]);  view_43 = None
        view_44: "bf16[128, 7, 7, 1024]" = torch.ops.aten.view.default(permute_13, _shape_param_45);  permute_13 = _shape_param_45 = None
        view_45: "bf16[128, 49, 1024]" = torch.ops.aten.view.default(view_44, _shape_param_46);  view_44 = _shape_param_46 = None
        view_46: "bf16[6272, 1024]" = torch.ops.aten.view.default(view_45, _shape_param_47);  view_45 = _shape_param_47 = None
        mm_13: "bf16[6272, 1024]" = torch.ops.aten.mm.default(view_46, arg45_1);  arg45_1 = None
        permute_14: "bf16[1024, 6272]" = torch.ops.aten.permute.default(view_46, [1, 0])
        mm_14: "bf16[1024, 1024]" = torch.ops.aten.mm.default(permute_14, arg46_1);  permute_14 = arg46_1 = None
        sum_25: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_46, [0], True, dtype = torch.float32);  view_46 = None
        view_47: "f32[1024]" = torch.ops.aten.view.default(sum_25, _shape_param_48);  sum_25 = _shape_param_48 = None
        convert_element_type_42: "bf16[1024]" = torch.ops.prims.convert_element_type.default(view_47, torch.bfloat16);  view_47 = None
        view_48: "bf16[128, 49, 1024]" = torch.ops.aten.view.default(mm_13, _shape_param_49);  mm_13 = _shape_param_49 = None
        convert_element_type_43: "f32[1024, 1024]" = torch.ops.prims.convert_element_type.default(mm_14, torch.float32);  mm_14 = None
        convert_element_type_44: "f32[1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_42, torch.float32);  convert_element_type_42 = None
        view_49: "bf16[128, 49, 32, 32]" = torch.ops.aten.view.default(view_48, _shape_param_50);  view_48 = _shape_param_50 = None
        permute_15: "bf16[128, 32, 49, 32]" = torch.ops.aten.permute.default(view_49, [0, 2, 1, 3]);  view_49 = None
        clone_3: "bf16[128, 32, 49, 32]" = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None
        view_50: "bf16[4096, 49, 32]" = torch.ops.aten.view.default(clone_3, _shape_param_51);  clone_3 = _shape_param_51 = None
        bmm_4: "bf16[4096, 49, 32]" = torch.ops.aten.bmm.default(arg47_1, view_50);  arg47_1 = None
        bmm_5: "bf16[4096, 49, 49]" = torch.ops.aten.bmm.default(view_50, arg48_1);  view_50 = arg48_1 = None
        view_51: "bf16[128, 32, 49, 32]" = torch.ops.aten.view.default(bmm_4, _shape_param_52);  bmm_4 = _shape_param_52 = None
        view_52: "bf16[128, 32, 49, 49]" = torch.ops.aten.view.default(bmm_5, _shape_param_53);  bmm_5 = _shape_param_53 = None
        convert_element_type_45: "f32[128, 32, 49, 49]" = torch.ops.prims.convert_element_type.default(view_52, torch.float32);  view_52 = None
        view_53: "bf16[128, 32, 49, 49]" = torch.ops.aten.view.default(arg49_1, _shape_param_54);  arg49_1 = _shape_param_54 = None
        view_54: "i64[2401]" = torch.ops.aten.view.default(arg50_1, [-1]);  arg50_1 = None
        index_1: "f32[2401, 32]" = torch.ops.aten.index.Tensor(arg51_1, [view_54]);  arg51_1 = None
        view_55: "f32[49, 49, 32]" = torch.ops.aten.view.default(index_1, _shape_param_55);  index_1 = _shape_param_55 = None
        permute_16: "f32[32, 49, 49]" = torch.ops.aten.permute.default(view_55, [2, 0, 1]);  view_55 = None
        clone_4: "f32[32, 49, 49]" = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
        unsqueeze_3: "f32[1, 32, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_4, 0);  clone_4 = None
        add_8: "f32[128, 32, 49, 49]" = torch.ops.aten.add.Tensor(view_53, unsqueeze_3);  view_53 = unsqueeze_3 = None
        sub_13: "f32[128, 32, 49, 49]" = torch.ops.aten.sub.Tensor(add_8, arg52_1);  add_8 = arg52_1 = None
        exp_3: "f32[128, 32, 49, 49]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
        div_10: "f32[128, 32, 49, 49]" = torch.ops.aten.div.Tensor(exp_3, arg53_1);  exp_3 = arg53_1 = None
        mul_48: "f32[128, 32, 49, 49]" = torch.ops.aten.mul.Tensor(convert_element_type_45, div_10);  convert_element_type_45 = None
        sum_26: "f32[128, 32, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_48, [-1], True)
        neg_1: "f32[128, 32, 49, 49]" = torch.ops.aten.neg.default(div_10);  div_10 = None
        fma_1: "f32[128, 32, 49, 49]" = torch.ops.prims.fma.default(neg_1, sum_26, mul_48);  neg_1 = sum_26 = mul_48 = None
        convert_element_type_46: "bf16[128, 32, 49, 49]" = torch.ops.prims.convert_element_type.default(fma_1, torch.bfloat16)
        sum_27: "f32[1, 32, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_1, [0], True, dtype = torch.float32);  fma_1 = None
        squeeze_1: "f32[32, 49, 49]" = torch.ops.aten.squeeze.dim(sum_27, 0);  sum_27 = None
        permute_17: "f32[49, 49, 32]" = torch.ops.aten.permute.default(squeeze_1, [1, 2, 0]);  squeeze_1 = None
        view_56: "f32[2401, 32]" = torch.ops.aten.view.default(permute_17, _shape_param_56);  permute_17 = _shape_param_56 = None
        index_put_1: "f32[169, 32]" = torch.ops.aten.index_put.default(full, [view_54], view_56, True);  full = view_54 = view_56 = None
        view_57: "bf16[4096, 49, 49]" = torch.ops.aten.view.default(convert_element_type_46, _shape_param_57);  convert_element_type_46 = _shape_param_57 = None
        bmm_6: "bf16[4096, 32, 49]" = torch.ops.aten.bmm.default(arg54_1, view_57);  arg54_1 = None
        bmm_7: "bf16[4096, 49, 32]" = torch.ops.aten.bmm.default(view_57, arg55_1);  view_57 = arg55_1 = None
        view_58: "bf16[128, 32, 32, 49]" = torch.ops.aten.view.default(bmm_6, _shape_param_58);  bmm_6 = _shape_param_58 = None
        view_59: "bf16[128, 32, 49, 32]" = torch.ops.aten.view.default(bmm_7, _shape_param_59);  bmm_7 = _shape_param_59 = None
        permute_18: "bf16[128, 32, 49, 32]" = torch.ops.aten.permute.default(view_58, [0, 1, 3, 2]);  view_58 = None
        mul_49: "bf16[128, 32, 49, 32]" = torch.ops.aten.mul.Tensor(view_59, 0.1767766952966369);  view_59 = None
        cat_1: "bf16[384, 32, 49, 32]" = torch.ops.aten.cat.default([mul_49, permute_18, view_51]);  mul_49 = permute_18 = view_51 = None
        view_60: "bf16[3, 128, 32, 49, 32]" = torch.ops.aten.view.default(cat_1, _shape_param_60);  cat_1 = _shape_param_60 = None
        permute_19: "bf16[128, 49, 3, 32, 32]" = torch.ops.aten.permute.default(view_60, [1, 3, 0, 2, 4]);  view_60 = None
        clone_5: "bf16[128, 49, 3, 32, 32]" = torch.ops.aten.clone.default(permute_19, memory_format = torch.contiguous_format);  permute_19 = None
        view_61: "bf16[128, 49, 3072]" = torch.ops.aten.view.default(clone_5, _shape_param_61);  clone_5 = _shape_param_61 = None
        view_62: "bf16[6272, 3072]" = torch.ops.aten.view.default(view_61, _shape_param_62);  view_61 = _shape_param_62 = None
        mm_15: "bf16[6272, 1024]" = torch.ops.aten.mm.default(view_62, arg56_1);  arg56_1 = None
        permute_20: "bf16[3072, 6272]" = torch.ops.aten.permute.default(view_62, [1, 0])
        mm_16: "bf16[3072, 1024]" = torch.ops.aten.mm.default(permute_20, arg57_1);  permute_20 = arg57_1 = None
        sum_28: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_62, [0], True, dtype = torch.float32);  view_62 = None
        view_63: "f32[3072]" = torch.ops.aten.view.default(sum_28, _shape_param_63);  sum_28 = _shape_param_63 = None
        convert_element_type_47: "bf16[3072]" = torch.ops.prims.convert_element_type.default(view_63, torch.bfloat16);  view_63 = None
        view_64: "bf16[128, 49, 1024]" = torch.ops.aten.view.default(mm_15, _shape_param_64);  mm_15 = _shape_param_64 = None
        convert_element_type_48: "f32[128, 49, 1024]" = torch.ops.prims.convert_element_type.default(view_64, torch.float32);  view_64 = None
        convert_element_type_49: "f32[3072, 1024]" = torch.ops.prims.convert_element_type.default(mm_16, torch.float32);  mm_16 = None
        convert_element_type_50: "f32[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_47, torch.float32);  convert_element_type_47 = None
        view_65: "f32[128, 7, 7, 1024]" = torch.ops.aten.view.default(convert_element_type_48, _shape_param_65);  convert_element_type_48 = _shape_param_65 = None
        view_66: "f32[128, 1, 1, 7, 7, 1024]" = torch.ops.aten.view.default(view_65, _shape_param_66);  view_65 = _shape_param_66 = None
        permute_21: "f32[128, 1, 7, 1, 7, 1024]" = torch.ops.aten.permute.default(view_66, [0, 1, 3, 2, 4, 5]);  view_66 = None
        view_67: "f32[128, 7, 7, 1024]" = torch.ops.aten.view.default(permute_21, _shape_param_67);  permute_21 = _shape_param_67 = None
        mul_50: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(view_67, arg58_1);  arg58_1 = None
        mul_51: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_50, 1024)
        sum_29: "f32[128, 7, 7, 1]" = torch.ops.aten.sum.dim_IntList(mul_50, [3], True)
        view_68: "bf16[128, 7, 7, 1024]" = torch.ops.aten.view.default(arg59_1, _shape_param_68);  arg59_1 = _shape_param_68 = None
        convert_element_type_51: "f32[128, 7, 7, 1024]" = torch.ops.prims.convert_element_type.default(view_68, torch.float32);  view_68 = None
        sub_14: "f32[128, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_51, arg60_1);  convert_element_type_51 = arg60_1 = None
        mul_52: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(sub_14, arg61_1);  sub_14 = None
        mul_53: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_50, mul_52);  mul_50 = None
        sum_30: "f32[128, 7, 7, 1]" = torch.ops.aten.sum.dim_IntList(mul_53, [3], True);  mul_53 = None
        mul_54: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_52, sum_30);  sum_30 = None
        sub_15: "f32[128, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(mul_51, sum_29);  mul_51 = sum_29 = None
        sub_16: "f32[128, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(sub_15, mul_54);  sub_15 = mul_54 = None
        div_11: "f32[128, 7, 7, 1]" = torch.ops.aten.div.Tensor(arg61_1, 1024);  arg61_1 = None
        mul_55: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(div_11, sub_16);  div_11 = sub_16 = None
        mul_56: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(view_67, mul_52);  mul_52 = None
        sum_31: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_56, [0, 1, 2]);  mul_56 = None
        sum_32: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_67, [0, 1, 2]);  view_67 = None
        convert_element_type_52: "bf16[128, 7, 7, 1024]" = torch.ops.prims.convert_element_type.default(mul_55, torch.bfloat16);  mul_55 = None
        add_9: "bf16[128, 7, 7, 1024]" = torch.ops.aten.add.Tensor(view_42, convert_element_type_52);  view_42 = convert_element_type_52 = None
        view_69: "bf16[6272, 1024]" = torch.ops.aten.view.default(add_9, _shape_param_69);  add_9 = _shape_param_69 = None
        permute_22: "bf16[1024, 6272]" = torch.ops.aten.permute.default(view_69, [1, 0])
        mm_17: "bf16[1024, 2048]" = torch.ops.aten.mm.default(permute_22, arg62_1);  permute_22 = arg62_1 = None
        mm_18: "bf16[6272, 2048]" = torch.ops.aten.mm.default(view_69, arg63_1);  view_69 = arg63_1 = None
        view_70: "bf16[128, 7, 7, 2048]" = torch.ops.aten.view.default(mm_18, _shape_param_70);  mm_18 = _shape_param_70 = None
        convert_element_type_53: "f32[128, 7, 7, 2048]" = torch.ops.prims.convert_element_type.default(view_70, torch.float32);  view_70 = None
        convert_element_type_54: "f32[1024, 2048]" = torch.ops.prims.convert_element_type.default(mm_17, torch.float32);  mm_17 = None
        mul_57: "f32[128, 7, 7, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_53, arg64_1);  arg64_1 = None
        mul_58: "f32[128, 7, 7, 2048]" = torch.ops.aten.mul.Tensor(mul_57, 2048)
        sum_33: "f32[128, 7, 7, 1]" = torch.ops.aten.sum.dim_IntList(mul_57, [3], True)
        convert_element_type_55: "f32[128, 7, 7, 2048]" = torch.ops.prims.convert_element_type.default(arg65_1, torch.float32);  arg65_1 = None
        sub_17: "f32[128, 7, 7, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_55, arg66_1);  convert_element_type_55 = arg66_1 = None
        mul_59: "f32[128, 7, 7, 2048]" = torch.ops.aten.mul.Tensor(sub_17, arg67_1);  sub_17 = None
        mul_60: "f32[128, 7, 7, 2048]" = torch.ops.aten.mul.Tensor(mul_57, mul_59);  mul_57 = None
        sum_34: "f32[128, 7, 7, 1]" = torch.ops.aten.sum.dim_IntList(mul_60, [3], True);  mul_60 = None
        mul_61: "f32[128, 7, 7, 2048]" = torch.ops.aten.mul.Tensor(mul_59, sum_34);  sum_34 = None
        sub_18: "f32[128, 7, 7, 2048]" = torch.ops.aten.sub.Tensor(mul_58, sum_33);  mul_58 = sum_33 = None
        sub_19: "f32[128, 7, 7, 2048]" = torch.ops.aten.sub.Tensor(sub_18, mul_61);  sub_18 = mul_61 = None
        div_12: "f32[128, 7, 7, 1]" = torch.ops.aten.div.Tensor(arg67_1, 2048);  arg67_1 = None
        mul_62: "f32[128, 7, 7, 2048]" = torch.ops.aten.mul.Tensor(div_12, sub_19);  div_12 = sub_19 = None
        mul_63: "f32[128, 7, 7, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_53, mul_59);  mul_59 = None
        sum_35: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_63, [0, 1, 2]);  mul_63 = None
        sum_36: "f32[2048]" = torch.ops.aten.sum.dim_IntList(convert_element_type_53, [0, 1, 2]);  convert_element_type_53 = None
        convert_element_type_56: "bf16[128, 7, 7, 2048]" = torch.ops.prims.convert_element_type.default(mul_62, torch.bfloat16);  mul_62 = None
        view_71: "bf16[128, 7, 7, 2, 2, 512]" = torch.ops.aten.view.default(convert_element_type_56, _shape_param_71);  convert_element_type_56 = _shape_param_71 = None
        permute_23: "bf16[128, 7, 2, 7, 2, 512]" = torch.ops.aten.permute.default(view_71, [0, 1, 4, 2, 3, 5]);  view_71 = None
        clone_6: "bf16[128, 7, 2, 7, 2, 512]" = torch.ops.aten.clone.default(permute_23, memory_format = torch.contiguous_format);  permute_23 = None
        view_72: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(clone_6, _shape_param_72);  clone_6 = _shape_param_72 = None
        view_73: "bf16[128, 196, 512]" = torch.ops.aten.view.default(view_72, _shape_param_73);  view_72 = _shape_param_73 = None
        convert_element_type_57: "bf16[128, 1, 1]" = torch.ops.prims.convert_element_type.default(arg68_1, torch.bfloat16);  arg68_1 = None
        div_13: "bf16[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_57, 0.9086956530809402);  convert_element_type_57 = None
        mul_64: "bf16[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_73, div_13);  div_13 = None
        view_74: "bf16[25088, 512]" = torch.ops.aten.view.default(mul_64, _shape_param_74);  mul_64 = _shape_param_74 = None
        mm_19: "bf16[25088, 2048]" = torch.ops.aten.mm.default(view_74, arg69_1);  arg69_1 = None
        permute_24: "bf16[512, 25088]" = torch.ops.aten.permute.default(view_74, [1, 0])
        mm_20: "bf16[512, 2048]" = torch.ops.aten.mm.default(permute_24, arg70_1);  permute_24 = arg70_1 = None
        sum_37: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_74, [0], True, dtype = torch.float32);  view_74 = None
        view_75: "f32[512]" = torch.ops.aten.view.default(sum_37, _shape_param_75);  sum_37 = _shape_param_75 = None
        convert_element_type_58: "bf16[512]" = torch.ops.prims.convert_element_type.default(view_75, torch.bfloat16);  view_75 = None
        view_76: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(mm_19, _shape_param_76);  mm_19 = _shape_param_76 = None
        convert_element_type_59: "f32[512, 2048]" = torch.ops.prims.convert_element_type.default(mm_20, torch.float32);  mm_20 = None
        convert_element_type_60: "f32[512]" = torch.ops.prims.convert_element_type.default(convert_element_type_58, torch.float32);  convert_element_type_58 = None
        convert_element_type_61: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_76, torch.float32);  view_76 = None
        view_77: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(arg71_1, _shape_param_77);  arg71_1 = _shape_param_77 = None
        convert_element_type_62: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_77, torch.float32);  view_77 = None
        mul_65: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_62, 0.7071067811865476)
        erf_2: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_65);  mul_65 = None
        add_10: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_66: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(add_10, 0.5);  add_10 = None
        mul_67: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_62, convert_element_type_62)
        mul_68: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_67, -0.5);  mul_67 = None
        exp_4: "f32[128, 196, 2048]" = torch.ops.aten.exp.default(mul_68);  mul_68 = None
        mul_69: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_4, 0.3989422804014327);  exp_4 = None
        mul_70: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_62, mul_69);  convert_element_type_62 = mul_69 = None
        add_11: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(mul_66, mul_70);  mul_66 = mul_70 = None
        mul_71: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_61, add_11);  convert_element_type_61 = add_11 = None
        convert_element_type_63: "bf16[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(mul_71, torch.bfloat16);  mul_71 = None
        view_78: "bf16[25088, 2048]" = torch.ops.aten.view.default(convert_element_type_63, _shape_param_78);  convert_element_type_63 = _shape_param_78 = None
        mm_21: "bf16[25088, 512]" = torch.ops.aten.mm.default(view_78, arg72_1);  arg72_1 = None
        permute_25: "bf16[2048, 25088]" = torch.ops.aten.permute.default(view_78, [1, 0])
        mm_22: "bf16[2048, 512]" = torch.ops.aten.mm.default(permute_25, arg73_1);  permute_25 = arg73_1 = None
        sum_38: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_78, [0], True, dtype = torch.float32);  view_78 = None
        view_79: "f32[2048]" = torch.ops.aten.view.default(sum_38, _shape_param_79);  sum_38 = _shape_param_79 = None
        convert_element_type_64: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_79, torch.bfloat16);  view_79 = None
        view_80: "bf16[128, 196, 512]" = torch.ops.aten.view.default(mm_21, _shape_param_80);  mm_21 = _shape_param_80 = None
        convert_element_type_65: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(view_80, torch.float32);  view_80 = None
        convert_element_type_66: "f32[2048, 512]" = torch.ops.prims.convert_element_type.default(mm_22, torch.float32);  mm_22 = None
        convert_element_type_67: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_64, torch.float32);  convert_element_type_64 = None
        mul_72: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_65, arg74_1);  arg74_1 = None
        mul_73: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_72, 512)
        sum_39: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_72, [2], True)
        convert_element_type_68: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(arg75_1, torch.float32);  arg75_1 = None
        sub_20: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_68, arg76_1);  convert_element_type_68 = arg76_1 = None
        mul_74: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_20, arg77_1);  sub_20 = None
        mul_75: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_72, mul_74);  mul_72 = None
        sum_40: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_75, [2], True);  mul_75 = None
        mul_76: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_74, sum_40);  sum_40 = None
        sub_21: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(mul_73, sum_39);  mul_73 = sum_39 = None
        sub_22: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(sub_21, mul_76);  sub_21 = mul_76 = None
        div_14: "f32[128, 196, 1]" = torch.ops.aten.div.Tensor(arg77_1, 512);  arg77_1 = None
        mul_77: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(div_14, sub_22);  div_14 = sub_22 = None
        mul_78: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_65, mul_74);  mul_74 = None
        sum_41: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_78, [0, 1]);  mul_78 = None
        sum_42: "f32[512]" = torch.ops.aten.sum.dim_IntList(convert_element_type_65, [0, 1]);  convert_element_type_65 = None
        convert_element_type_69: "bf16[128, 196, 512]" = torch.ops.prims.convert_element_type.default(mul_77, torch.bfloat16);  mul_77 = None
        add_12: "bf16[128, 196, 512]" = torch.ops.aten.add.Tensor(view_73, convert_element_type_69);  view_73 = convert_element_type_69 = None
        view_81: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(add_12, _shape_param_81);  add_12 = _shape_param_81 = None
        convert_element_type_70: "bf16[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(arg78_1, torch.bfloat16);  arg78_1 = None
        div_15: "bf16[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_70, 0.9086956530809402);  convert_element_type_70 = None
        mul_79: "bf16[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_81, div_15);  div_15 = None
        iota: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_13: "i64[14]" = torch.ops.aten.add.Tensor(iota, 3);  iota = None
        fmod: "i64[14]" = torch.ops.aten.fmod.Scalar(add_13, 14);  add_13 = None
        index_2: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(mul_79, [None, None, fmod]);  mul_79 = None
        index_3: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_2, [None, fmod]);  index_2 = None
        view_82: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(index_3, _shape_param_82);  index_3 = _shape_param_82 = None
        permute_26: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_82, [0, 1, 3, 2, 4, 5]);  view_82 = None
        clone_7: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_26, memory_format = torch.contiguous_format);  permute_26 = None
        view_83: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(clone_7, _shape_param_83);  clone_7 = _shape_param_83 = None
        view_84: "bf16[512, 49, 512]" = torch.ops.aten.view.default(view_83, _shape_param_84);  view_83 = _shape_param_84 = None
        view_85: "bf16[25088, 512]" = torch.ops.aten.view.default(view_84, _shape_param_85);  view_84 = _shape_param_85 = None
        mm_23: "bf16[25088, 512]" = torch.ops.aten.mm.default(view_85, arg79_1);  arg79_1 = None
        permute_27: "bf16[512, 25088]" = torch.ops.aten.permute.default(view_85, [1, 0])
        mm_24: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_27, arg80_1);  permute_27 = arg80_1 = None
        sum_43: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_85, [0], True, dtype = torch.float32);  view_85 = None
        view_86: "f32[512]" = torch.ops.aten.view.default(sum_43, _shape_param_86);  sum_43 = _shape_param_86 = None
        convert_element_type_71: "bf16[512]" = torch.ops.prims.convert_element_type.default(view_86, torch.bfloat16);  view_86 = None
        view_87: "bf16[512, 49, 512]" = torch.ops.aten.view.default(mm_23, _shape_param_87);  mm_23 = _shape_param_87 = None
        convert_element_type_72: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_24, torch.float32);  mm_24 = None
        convert_element_type_73: "f32[512]" = torch.ops.prims.convert_element_type.default(convert_element_type_71, torch.float32);  convert_element_type_71 = None
        view_88: "bf16[512, 49, 16, 32]" = torch.ops.aten.view.default(view_87, _shape_param_88);  view_87 = _shape_param_88 = None
        permute_28: "bf16[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_88, [0, 2, 1, 3]);  view_88 = None
        clone_8: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(permute_28, memory_format = torch.contiguous_format);  permute_28 = None
        view_89: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_8, _shape_param_89);  clone_8 = _shape_param_89 = None
        bmm_8: "bf16[8192, 49, 32]" = torch.ops.aten.bmm.default(arg81_1, view_89);  arg81_1 = None
        bmm_9: "bf16[8192, 49, 49]" = torch.ops.aten.bmm.default(view_89, arg82_1);  view_89 = arg82_1 = None
        view_90: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(bmm_8, _shape_param_90);  bmm_8 = _shape_param_90 = None
        view_91: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(bmm_9, _shape_param_91);  bmm_9 = _shape_param_91 = None
        convert_element_type_74: "f32[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(view_91, torch.float32);  view_91 = None
        slice_1: "bf16[8192, 49, 56]" = torch.ops.aten.slice.Tensor(arg83_1, 1, 0, -7);  arg83_1 = None
        slice_2: "bf16[8192, 49, 49]" = torch.ops.aten.slice.Tensor(slice_1, 2, 0, -7);  slice_1 = None
        view_92: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(slice_2, _shape_param_92);  slice_2 = _shape_param_92 = None
        view_93: "i64[2401]" = torch.ops.aten.view.default(arg84_1, [-1]);  arg84_1 = None
        index_4: "f32[2401, 16]" = torch.ops.aten.index.Tensor(arg85_1, [view_93]);  arg85_1 = None
        view_94: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_4, _shape_param_93);  index_4 = _shape_param_93 = None
        permute_29: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_94, [2, 0, 1]);  view_94 = None
        clone_9: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
        unsqueeze_4: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_9, 0);  clone_9 = None
        add_14: "f32[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_92, unsqueeze_4);  view_92 = unsqueeze_4 = None
        view_95: "f32[128, 4, 16, 49, 49]" = torch.ops.aten.view.default(add_14, _shape_param_94);  add_14 = _shape_param_94 = None
        unsqueeze_5: "f32[4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(arg86_1, 1);  arg86_1 = None
        unsqueeze_6: "f32[1, 4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_5, 0);  unsqueeze_5 = None
        add_15: "f32[128, 4, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_95, unsqueeze_6);  view_95 = unsqueeze_6 = None
        view_96: "f32[512, 16, 49, 49]" = torch.ops.aten.view.default(add_15, _shape_param_95);  add_15 = _shape_param_95 = None
        sub_23: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(view_96, arg87_1);  view_96 = arg87_1 = None
        exp_5: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_23);  sub_23 = None
        div_16: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_5, arg88_1);  exp_5 = arg88_1 = None
        mul_80: "f32[512, 16, 49, 49]" = torch.ops.aten.mul.Tensor(convert_element_type_74, div_16);  convert_element_type_74 = None
        sum_44: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_80, [-1], True)
        neg_2: "f32[512, 16, 49, 49]" = torch.ops.aten.neg.default(div_16);  div_16 = None
        fma_2: "f32[512, 16, 49, 49]" = torch.ops.prims.fma.default(neg_2, sum_44, mul_80);  neg_2 = sum_44 = mul_80 = None
        convert_element_type_75: "bf16[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(fma_2, torch.bfloat16)
        sum_45: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_2, [0], True, dtype = torch.float32);  fma_2 = None
        squeeze_2: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_45, 0);  sum_45 = None
        permute_30: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_2, [1, 2, 0]);  squeeze_2 = None
        view_97: "f32[2401, 16]" = torch.ops.aten.view.default(permute_30, _shape_param_96);  permute_30 = _shape_param_96 = None
        full_1: "f32[169, 16]" = torch.ops.aten.full.default(_shape_param_97, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_97 = None
        index_put_2: "f32[169, 16]" = torch.ops.aten.index_put.default(full_1, [view_93], view_97, True);  view_93 = view_97 = None
        view_98: "bf16[8192, 49, 49]" = torch.ops.aten.view.default(convert_element_type_75, _shape_param_98);  convert_element_type_75 = _shape_param_98 = None
        bmm_10: "bf16[8192, 32, 49]" = torch.ops.aten.bmm.default(arg89_1, view_98);  arg89_1 = None
        bmm_11: "bf16[8192, 49, 32]" = torch.ops.aten.bmm.default(view_98, arg90_1);  view_98 = arg90_1 = None
        view_99: "bf16[512, 16, 32, 49]" = torch.ops.aten.view.default(bmm_10, _shape_param_99);  bmm_10 = _shape_param_99 = None
        view_100: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(bmm_11, _shape_param_100);  bmm_11 = _shape_param_100 = None
        permute_31: "bf16[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_99, [0, 1, 3, 2]);  view_99 = None
        mul_81: "bf16[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_100, 0.1767766952966369);  view_100 = None
        cat_2: "bf16[1536, 16, 49, 32]" = torch.ops.aten.cat.default([mul_81, permute_31, view_90]);  mul_81 = permute_31 = view_90 = None
        view_101: "bf16[3, 512, 16, 49, 32]" = torch.ops.aten.view.default(cat_2, _shape_param_101);  cat_2 = _shape_param_101 = None
        permute_32: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.permute.default(view_101, [1, 3, 0, 2, 4]);  view_101 = None
        clone_10: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_32, memory_format = torch.contiguous_format);  permute_32 = None
        view_102: "bf16[512, 49, 1536]" = torch.ops.aten.view.default(clone_10, _shape_param_102);  clone_10 = _shape_param_102 = None
        view_103: "bf16[25088, 1536]" = torch.ops.aten.view.default(view_102, _shape_param_103);  view_102 = _shape_param_103 = None
        mm_25: "bf16[25088, 512]" = torch.ops.aten.mm.default(view_103, arg91_1);  arg91_1 = None
        permute_33: "bf16[1536, 25088]" = torch.ops.aten.permute.default(view_103, [1, 0])
        mm_26: "bf16[1536, 512]" = torch.ops.aten.mm.default(permute_33, arg92_1);  permute_33 = arg92_1 = None
        sum_46: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_103, [0], True, dtype = torch.float32);  view_103 = None
        view_104: "f32[1536]" = torch.ops.aten.view.default(sum_46, _shape_param_104);  sum_46 = _shape_param_104 = None
        convert_element_type_76: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_104, torch.bfloat16);  view_104 = None
        view_105: "bf16[512, 49, 512]" = torch.ops.aten.view.default(mm_25, _shape_param_105);  mm_25 = _shape_param_105 = None
        convert_element_type_77: "f32[512, 49, 512]" = torch.ops.prims.convert_element_type.default(view_105, torch.float32);  view_105 = None
        convert_element_type_78: "f32[1536, 512]" = torch.ops.prims.convert_element_type.default(mm_26, torch.float32);  mm_26 = None
        convert_element_type_79: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_76, torch.float32);  convert_element_type_76 = None
        view_106: "f32[512, 7, 7, 512]" = torch.ops.aten.view.default(convert_element_type_77, _shape_param_106);  convert_element_type_77 = _shape_param_106 = None
        view_107: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_106, _shape_param_107);  view_106 = _shape_param_107 = None
        permute_34: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_107, [0, 1, 3, 2, 4, 5]);  view_107 = None
        clone_11: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_34, memory_format = torch.contiguous_format);  permute_34 = None
        view_108: "f32[128, 14, 14, 512]" = torch.ops.aten.view.default(clone_11, _shape_param_108);  clone_11 = _shape_param_108 = None
        index_5: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(view_108, [None, None, arg93_1]);  view_108 = None
        index_6: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_5, [None, arg93_1]);  index_5 = None
        mul_82: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(index_6, arg94_1);  arg94_1 = None
        mul_83: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_82, 512)
        sum_47: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_82, [3], True)
        convert_element_type_80: "f32[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(arg95_1, torch.float32);  arg95_1 = None
        sub_24: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_80, arg96_1);  convert_element_type_80 = arg96_1 = None
        mul_84: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_24, arg97_1);  sub_24 = None
        mul_85: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_82, mul_84);  mul_82 = None
        sum_48: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_85, [3], True);  mul_85 = None
        mul_86: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_84, sum_48);  sum_48 = None
        sub_25: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_83, sum_47);  mul_83 = sum_47 = None
        sub_26: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_25, mul_86);  sub_25 = mul_86 = None
        div_17: "f32[128, 14, 14, 1]" = torch.ops.aten.div.Tensor(arg97_1, 512);  arg97_1 = None
        mul_87: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_17, sub_26);  div_17 = sub_26 = None
        mul_88: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(index_6, mul_84);  mul_84 = None
        sum_49: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_88, [0, 1, 2]);  mul_88 = None
        sum_50: "f32[512]" = torch.ops.aten.sum.dim_IntList(index_6, [0, 1, 2]);  index_6 = None
        convert_element_type_81: "bf16[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(mul_87, torch.bfloat16);  mul_87 = None
        add_16: "bf16[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_81, convert_element_type_81);  view_81 = convert_element_type_81 = None
        view_109: "bf16[128, 196, 512]" = torch.ops.aten.view.default(add_16, _shape_param_109);  add_16 = _shape_param_109 = None
        convert_element_type_82: "bf16[128, 1, 1]" = torch.ops.prims.convert_element_type.default(arg98_1, torch.bfloat16);  arg98_1 = None
        div_18: "bf16[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_82, 0.9130434766411781);  convert_element_type_82 = None
        mul_89: "bf16[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_109, div_18);  div_18 = None
        view_110: "bf16[25088, 512]" = torch.ops.aten.view.default(mul_89, _shape_param_110);  mul_89 = _shape_param_110 = None
        mm_27: "bf16[25088, 2048]" = torch.ops.aten.mm.default(view_110, arg99_1);  arg99_1 = None
        permute_35: "bf16[512, 25088]" = torch.ops.aten.permute.default(view_110, [1, 0])
        mm_28: "bf16[512, 2048]" = torch.ops.aten.mm.default(permute_35, arg100_1);  permute_35 = arg100_1 = None
        sum_51: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_110, [0], True, dtype = torch.float32);  view_110 = None
        view_111: "f32[512]" = torch.ops.aten.view.default(sum_51, _shape_param_111);  sum_51 = _shape_param_111 = None
        convert_element_type_83: "bf16[512]" = torch.ops.prims.convert_element_type.default(view_111, torch.bfloat16);  view_111 = None
        view_112: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(mm_27, _shape_param_112);  mm_27 = _shape_param_112 = None
        convert_element_type_84: "f32[512, 2048]" = torch.ops.prims.convert_element_type.default(mm_28, torch.float32);  mm_28 = None
        convert_element_type_85: "f32[512]" = torch.ops.prims.convert_element_type.default(convert_element_type_83, torch.float32);  convert_element_type_83 = None
        convert_element_type_86: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_112, torch.float32);  view_112 = None
        view_113: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(arg101_1, _shape_param_113);  arg101_1 = _shape_param_113 = None
        convert_element_type_87: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_113, torch.float32);  view_113 = None
        mul_90: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_87, 0.7071067811865476)
        erf_3: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_90);  mul_90 = None
        add_17: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_91: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(add_17, 0.5);  add_17 = None
        mul_92: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_87, convert_element_type_87)
        mul_93: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_92, -0.5);  mul_92 = None
        exp_6: "f32[128, 196, 2048]" = torch.ops.aten.exp.default(mul_93);  mul_93 = None
        mul_94: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_6, 0.3989422804014327);  exp_6 = None
        mul_95: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_87, mul_94);  convert_element_type_87 = mul_94 = None
        add_18: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(mul_91, mul_95);  mul_91 = mul_95 = None
        mul_96: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_86, add_18);  convert_element_type_86 = add_18 = None
        convert_element_type_88: "bf16[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(mul_96, torch.bfloat16);  mul_96 = None
        view_114: "bf16[25088, 2048]" = torch.ops.aten.view.default(convert_element_type_88, _shape_param_114);  convert_element_type_88 = _shape_param_114 = None
        mm_29: "bf16[25088, 512]" = torch.ops.aten.mm.default(view_114, arg102_1);  arg102_1 = None
        permute_36: "bf16[2048, 25088]" = torch.ops.aten.permute.default(view_114, [1, 0])
        mm_30: "bf16[2048, 512]" = torch.ops.aten.mm.default(permute_36, arg103_1);  permute_36 = arg103_1 = None
        sum_52: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_114, [0], True, dtype = torch.float32);  view_114 = None
        view_115: "f32[2048]" = torch.ops.aten.view.default(sum_52, _shape_param_115);  sum_52 = _shape_param_115 = None
        convert_element_type_89: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_115, torch.bfloat16);  view_115 = None
        view_116: "bf16[128, 196, 512]" = torch.ops.aten.view.default(mm_29, _shape_param_116);  mm_29 = _shape_param_116 = None
        convert_element_type_90: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(view_116, torch.float32);  view_116 = None
        convert_element_type_91: "f32[2048, 512]" = torch.ops.prims.convert_element_type.default(mm_30, torch.float32);  mm_30 = None
        convert_element_type_92: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_89, torch.float32);  convert_element_type_89 = None
        mul_97: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_90, arg104_1);  arg104_1 = None
        mul_98: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_97, 512)
        sum_53: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_97, [2], True)
        convert_element_type_93: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(arg105_1, torch.float32);  arg105_1 = None
        sub_27: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_93, arg106_1);  convert_element_type_93 = arg106_1 = None
        mul_99: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_27, arg107_1);  sub_27 = None
        mul_100: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_97, mul_99);  mul_97 = None
        sum_54: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_100, [2], True);  mul_100 = None
        mul_101: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_99, sum_54);  sum_54 = None
        sub_28: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(mul_98, sum_53);  mul_98 = sum_53 = None
        sub_29: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(sub_28, mul_101);  sub_28 = mul_101 = None
        div_19: "f32[128, 196, 1]" = torch.ops.aten.div.Tensor(arg107_1, 512);  arg107_1 = None
        mul_102: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(div_19, sub_29);  div_19 = sub_29 = None
        mul_103: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_90, mul_99);  mul_99 = None
        sum_55: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_103, [0, 1]);  mul_103 = None
        sum_56: "f32[512]" = torch.ops.aten.sum.dim_IntList(convert_element_type_90, [0, 1]);  convert_element_type_90 = None
        convert_element_type_94: "bf16[128, 196, 512]" = torch.ops.prims.convert_element_type.default(mul_102, torch.bfloat16);  mul_102 = None
        add_19: "bf16[128, 196, 512]" = torch.ops.aten.add.Tensor(view_109, convert_element_type_94);  view_109 = convert_element_type_94 = None
        view_117: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(add_19, _shape_param_117);  add_19 = _shape_param_117 = None
        convert_element_type_95: "bf16[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(arg108_1, torch.bfloat16);  arg108_1 = None
        div_20: "bf16[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_95, 0.9130434766411781);  convert_element_type_95 = None
        mul_104: "bf16[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_117, div_20);  div_20 = None
        view_118: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(mul_104, _shape_param_118);  mul_104 = _shape_param_118 = None
        permute_37: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_118, [0, 1, 3, 2, 4, 5]);  view_118 = None
        clone_12: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None
        view_119: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(clone_12, _shape_param_119);  clone_12 = _shape_param_119 = None
        view_120: "bf16[512, 49, 512]" = torch.ops.aten.view.default(view_119, _shape_param_120);  view_119 = _shape_param_120 = None
        view_121: "bf16[25088, 512]" = torch.ops.aten.view.default(view_120, _shape_param_121);  view_120 = _shape_param_121 = None
        mm_31: "bf16[25088, 512]" = torch.ops.aten.mm.default(view_121, arg109_1);  arg109_1 = None
        permute_38: "bf16[512, 25088]" = torch.ops.aten.permute.default(view_121, [1, 0])
        mm_32: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_38, arg110_1);  permute_38 = arg110_1 = None
        sum_57: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_121, [0], True, dtype = torch.float32);  view_121 = None
        view_122: "f32[512]" = torch.ops.aten.view.default(sum_57, _shape_param_122);  sum_57 = _shape_param_122 = None
        convert_element_type_96: "bf16[512]" = torch.ops.prims.convert_element_type.default(view_122, torch.bfloat16);  view_122 = None
        view_123: "bf16[512, 49, 512]" = torch.ops.aten.view.default(mm_31, _shape_param_123);  mm_31 = _shape_param_123 = None
        convert_element_type_97: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_32, torch.float32);  mm_32 = None
        convert_element_type_98: "f32[512]" = torch.ops.prims.convert_element_type.default(convert_element_type_96, torch.float32);  convert_element_type_96 = None
        view_124: "bf16[512, 49, 16, 32]" = torch.ops.aten.view.default(view_123, _shape_param_124);  view_123 = _shape_param_124 = None
        permute_39: "bf16[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_124, [0, 2, 1, 3]);  view_124 = None
        clone_13: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(permute_39, memory_format = torch.contiguous_format);  permute_39 = None
        view_125: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_13, _shape_param_125);  clone_13 = _shape_param_125 = None
        bmm_12: "bf16[8192, 49, 32]" = torch.ops.aten.bmm.default(arg111_1, view_125);  arg111_1 = None
        bmm_13: "bf16[8192, 49, 49]" = torch.ops.aten.bmm.default(view_125, arg112_1);  view_125 = arg112_1 = None
        view_126: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(bmm_12, _shape_param_126);  bmm_12 = _shape_param_126 = None
        view_127: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(bmm_13, _shape_param_127);  bmm_13 = _shape_param_127 = None
        convert_element_type_99: "f32[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(view_127, torch.float32);  view_127 = None
        slice_3: "bf16[8192, 49, 56]" = torch.ops.aten.slice.Tensor(arg113_1, 1, 0, -7);  arg113_1 = None
        slice_4: "bf16[8192, 49, 49]" = torch.ops.aten.slice.Tensor(slice_3, 2, 0, -7);  slice_3 = None
        view_128: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(slice_4, _shape_param_128);  slice_4 = _shape_param_128 = None
        view_129: "i64[2401]" = torch.ops.aten.view.default(arg114_1, [-1]);  arg114_1 = None
        index_7: "f32[2401, 16]" = torch.ops.aten.index.Tensor(arg115_1, [view_129]);  arg115_1 = None
        view_130: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_7, _shape_param_129);  index_7 = _shape_param_129 = None
        permute_40: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_130, [2, 0, 1]);  view_130 = None
        clone_14: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
        unsqueeze_7: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_14, 0);  clone_14 = None
        add_20: "f32[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_128, unsqueeze_7);  view_128 = unsqueeze_7 = None
        sub_30: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(add_20, arg116_1);  add_20 = arg116_1 = None
        exp_7: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_30);  sub_30 = None
        div_21: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_7, arg117_1);  exp_7 = arg117_1 = None
        mul_105: "f32[512, 16, 49, 49]" = torch.ops.aten.mul.Tensor(convert_element_type_99, div_21);  convert_element_type_99 = None
        sum_58: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_105, [-1], True)
        neg_3: "f32[512, 16, 49, 49]" = torch.ops.aten.neg.default(div_21);  div_21 = None
        fma_3: "f32[512, 16, 49, 49]" = torch.ops.prims.fma.default(neg_3, sum_58, mul_105);  neg_3 = sum_58 = mul_105 = None
        convert_element_type_100: "bf16[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(fma_3, torch.bfloat16)
        sum_59: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_3, [0], True, dtype = torch.float32);  fma_3 = None
        squeeze_3: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_59, 0);  sum_59 = None
        permute_41: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_3, [1, 2, 0]);  squeeze_3 = None
        view_131: "f32[2401, 16]" = torch.ops.aten.view.default(permute_41, _shape_param_130);  permute_41 = _shape_param_130 = None
        index_put_3: "f32[169, 16]" = torch.ops.aten.index_put.default(full_1, [view_129], view_131, True);  view_129 = view_131 = None
        view_132: "bf16[8192, 49, 49]" = torch.ops.aten.view.default(convert_element_type_100, _shape_param_131);  convert_element_type_100 = _shape_param_131 = None
        bmm_14: "bf16[8192, 32, 49]" = torch.ops.aten.bmm.default(arg118_1, view_132);  arg118_1 = None
        bmm_15: "bf16[8192, 49, 32]" = torch.ops.aten.bmm.default(view_132, arg119_1);  view_132 = arg119_1 = None
        view_133: "bf16[512, 16, 32, 49]" = torch.ops.aten.view.default(bmm_14, _shape_param_132);  bmm_14 = _shape_param_132 = None
        view_134: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(bmm_15, _shape_param_133);  bmm_15 = _shape_param_133 = None
        permute_42: "bf16[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_133, [0, 1, 3, 2]);  view_133 = None
        mul_106: "bf16[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_134, 0.1767766952966369);  view_134 = None
        cat_3: "bf16[1536, 16, 49, 32]" = torch.ops.aten.cat.default([mul_106, permute_42, view_126]);  mul_106 = permute_42 = view_126 = None
        view_135: "bf16[3, 512, 16, 49, 32]" = torch.ops.aten.view.default(cat_3, _shape_param_134);  cat_3 = _shape_param_134 = None
        permute_43: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.permute.default(view_135, [1, 3, 0, 2, 4]);  view_135 = None
        clone_15: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_43, memory_format = torch.contiguous_format);  permute_43 = None
        view_136: "bf16[512, 49, 1536]" = torch.ops.aten.view.default(clone_15, _shape_param_135);  clone_15 = _shape_param_135 = None
        view_137: "bf16[25088, 1536]" = torch.ops.aten.view.default(view_136, _shape_param_136);  view_136 = _shape_param_136 = None
        mm_33: "bf16[25088, 512]" = torch.ops.aten.mm.default(view_137, arg120_1);  arg120_1 = None
        permute_44: "bf16[1536, 25088]" = torch.ops.aten.permute.default(view_137, [1, 0])
        mm_34: "bf16[1536, 512]" = torch.ops.aten.mm.default(permute_44, arg121_1);  permute_44 = arg121_1 = None
        sum_60: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_137, [0], True, dtype = torch.float32);  view_137 = None
        view_138: "f32[1536]" = torch.ops.aten.view.default(sum_60, _shape_param_137);  sum_60 = _shape_param_137 = None
        convert_element_type_101: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_138, torch.bfloat16);  view_138 = None
        view_139: "bf16[512, 49, 512]" = torch.ops.aten.view.default(mm_33, _shape_param_138);  mm_33 = _shape_param_138 = None
        convert_element_type_102: "f32[512, 49, 512]" = torch.ops.prims.convert_element_type.default(view_139, torch.float32);  view_139 = None
        convert_element_type_103: "f32[1536, 512]" = torch.ops.prims.convert_element_type.default(mm_34, torch.float32);  mm_34 = None
        convert_element_type_104: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_101, torch.float32);  convert_element_type_101 = None
        view_140: "f32[512, 7, 7, 512]" = torch.ops.aten.view.default(convert_element_type_102, _shape_param_139);  convert_element_type_102 = _shape_param_139 = None
        view_141: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_140, _shape_param_140);  view_140 = _shape_param_140 = None
        permute_45: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_141, [0, 1, 3, 2, 4, 5]);  view_141 = None
        clone_16: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_45, memory_format = torch.contiguous_format);  permute_45 = None
        view_142: "f32[128, 14, 14, 512]" = torch.ops.aten.view.default(clone_16, _shape_param_141);  clone_16 = _shape_param_141 = None
        mul_107: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_142, arg122_1);  arg122_1 = None
        mul_108: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_107, 512)
        sum_61: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_107, [3], True)
        convert_element_type_105: "f32[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(arg123_1, torch.float32);  arg123_1 = None
        sub_31: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_105, arg124_1);  convert_element_type_105 = arg124_1 = None
        mul_109: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_31, arg125_1);  sub_31 = None
        mul_110: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_107, mul_109);  mul_107 = None
        sum_62: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_110, [3], True);  mul_110 = None
        mul_111: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_109, sum_62);  sum_62 = None
        sub_32: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_108, sum_61);  mul_108 = sum_61 = None
        sub_33: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_32, mul_111);  sub_32 = mul_111 = None
        div_22: "f32[128, 14, 14, 1]" = torch.ops.aten.div.Tensor(arg125_1, 512);  arg125_1 = None
        mul_112: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_22, sub_33);  div_22 = sub_33 = None
        mul_113: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_142, mul_109);  mul_109 = None
        sum_63: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_113, [0, 1, 2]);  mul_113 = None
        sum_64: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_142, [0, 1, 2]);  view_142 = None
        convert_element_type_106: "bf16[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(mul_112, torch.bfloat16);  mul_112 = None
        add_21: "bf16[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_117, convert_element_type_106);  view_117 = convert_element_type_106 = None
        view_143: "bf16[128, 196, 512]" = torch.ops.aten.view.default(add_21, _shape_param_142);  add_21 = _shape_param_142 = None
        convert_element_type_107: "bf16[128, 1, 1]" = torch.ops.prims.convert_element_type.default(arg126_1, torch.bfloat16);  arg126_1 = None
        div_23: "bf16[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_107, 0.917391300201416);  convert_element_type_107 = None
        mul_114: "bf16[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_143, div_23);  div_23 = None
        view_144: "bf16[25088, 512]" = torch.ops.aten.view.default(mul_114, _shape_param_143);  mul_114 = _shape_param_143 = None
        mm_35: "bf16[25088, 2048]" = torch.ops.aten.mm.default(view_144, arg127_1);  arg127_1 = None
        permute_46: "bf16[512, 25088]" = torch.ops.aten.permute.default(view_144, [1, 0])
        mm_36: "bf16[512, 2048]" = torch.ops.aten.mm.default(permute_46, arg128_1);  permute_46 = arg128_1 = None
        sum_65: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_144, [0], True, dtype = torch.float32);  view_144 = None
        view_145: "f32[512]" = torch.ops.aten.view.default(sum_65, _shape_param_144);  sum_65 = _shape_param_144 = None
        convert_element_type_108: "bf16[512]" = torch.ops.prims.convert_element_type.default(view_145, torch.bfloat16);  view_145 = None
        view_146: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(mm_35, _shape_param_145);  mm_35 = _shape_param_145 = None
        convert_element_type_109: "f32[512, 2048]" = torch.ops.prims.convert_element_type.default(mm_36, torch.float32);  mm_36 = None
        convert_element_type_110: "f32[512]" = torch.ops.prims.convert_element_type.default(convert_element_type_108, torch.float32);  convert_element_type_108 = None
        convert_element_type_111: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_146, torch.float32);  view_146 = None
        view_147: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(arg129_1, _shape_param_146);  arg129_1 = _shape_param_146 = None
        convert_element_type_112: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_147, torch.float32);  view_147 = None
        mul_115: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_112, 0.7071067811865476)
        erf_4: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_115);  mul_115 = None
        add_22: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_116: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(add_22, 0.5);  add_22 = None
        mul_117: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_112, convert_element_type_112)
        mul_118: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_117, -0.5);  mul_117 = None
        exp_8: "f32[128, 196, 2048]" = torch.ops.aten.exp.default(mul_118);  mul_118 = None
        mul_119: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_8, 0.3989422804014327);  exp_8 = None
        mul_120: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_112, mul_119);  convert_element_type_112 = mul_119 = None
        add_23: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(mul_116, mul_120);  mul_116 = mul_120 = None
        mul_121: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_111, add_23);  convert_element_type_111 = add_23 = None
        convert_element_type_113: "bf16[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(mul_121, torch.bfloat16);  mul_121 = None
        view_148: "bf16[25088, 2048]" = torch.ops.aten.view.default(convert_element_type_113, _shape_param_147);  convert_element_type_113 = _shape_param_147 = None
        mm_37: "bf16[25088, 512]" = torch.ops.aten.mm.default(view_148, arg130_1);  arg130_1 = None
        permute_47: "bf16[2048, 25088]" = torch.ops.aten.permute.default(view_148, [1, 0])
        mm_38: "bf16[2048, 512]" = torch.ops.aten.mm.default(permute_47, arg131_1);  permute_47 = arg131_1 = None
        sum_66: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_148, [0], True, dtype = torch.float32);  view_148 = None
        view_149: "f32[2048]" = torch.ops.aten.view.default(sum_66, _shape_param_148);  sum_66 = _shape_param_148 = None
        convert_element_type_114: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_149, torch.bfloat16);  view_149 = None
        view_150: "bf16[128, 196, 512]" = torch.ops.aten.view.default(mm_37, _shape_param_149);  mm_37 = _shape_param_149 = None
        convert_element_type_115: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(view_150, torch.float32);  view_150 = None
        convert_element_type_116: "f32[2048, 512]" = torch.ops.prims.convert_element_type.default(mm_38, torch.float32);  mm_38 = None
        convert_element_type_117: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_114, torch.float32);  convert_element_type_114 = None
        mul_122: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_115, arg132_1);  arg132_1 = None
        mul_123: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_122, 512)
        sum_67: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_122, [2], True)
        convert_element_type_118: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(arg133_1, torch.float32);  arg133_1 = None
        sub_34: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_118, arg134_1);  convert_element_type_118 = arg134_1 = None
        mul_124: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_34, arg135_1);  sub_34 = None
        mul_125: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_122, mul_124);  mul_122 = None
        sum_68: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_125, [2], True);  mul_125 = None
        mul_126: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_124, sum_68);  sum_68 = None
        sub_35: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(mul_123, sum_67);  mul_123 = sum_67 = None
        sub_36: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(sub_35, mul_126);  sub_35 = mul_126 = None
        div_24: "f32[128, 196, 1]" = torch.ops.aten.div.Tensor(arg135_1, 512);  arg135_1 = None
        mul_127: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(div_24, sub_36);  div_24 = sub_36 = None
        mul_128: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_115, mul_124);  mul_124 = None
        sum_69: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_128, [0, 1]);  mul_128 = None
        sum_70: "f32[512]" = torch.ops.aten.sum.dim_IntList(convert_element_type_115, [0, 1]);  convert_element_type_115 = None
        convert_element_type_119: "bf16[128, 196, 512]" = torch.ops.prims.convert_element_type.default(mul_127, torch.bfloat16);  mul_127 = None
        add_24: "bf16[128, 196, 512]" = torch.ops.aten.add.Tensor(view_143, convert_element_type_119);  view_143 = convert_element_type_119 = None
        view_151: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(add_24, _shape_param_150);  add_24 = _shape_param_150 = None
        convert_element_type_120: "bf16[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(arg136_1, torch.bfloat16);  arg136_1 = None
        div_25: "bf16[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_120, 0.917391300201416);  convert_element_type_120 = None
        mul_129: "bf16[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_151, div_25);  div_25 = None
        index_8: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(mul_129, [None, None, fmod]);  mul_129 = None
        index_9: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_8, [None, fmod]);  index_8 = None
        view_152: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(index_9, _shape_param_151);  index_9 = _shape_param_151 = None
        permute_48: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_152, [0, 1, 3, 2, 4, 5]);  view_152 = None
        clone_17: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_48, memory_format = torch.contiguous_format);  permute_48 = None
        view_153: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(clone_17, _shape_param_152);  clone_17 = _shape_param_152 = None
        view_154: "bf16[512, 49, 512]" = torch.ops.aten.view.default(view_153, _shape_param_153);  view_153 = _shape_param_153 = None
        view_155: "bf16[25088, 512]" = torch.ops.aten.view.default(view_154, _shape_param_154);  view_154 = _shape_param_154 = None
        mm_39: "bf16[25088, 512]" = torch.ops.aten.mm.default(view_155, arg137_1);  arg137_1 = None
        permute_49: "bf16[512, 25088]" = torch.ops.aten.permute.default(view_155, [1, 0])
        mm_40: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_49, arg138_1);  permute_49 = arg138_1 = None
        sum_71: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_155, [0], True, dtype = torch.float32);  view_155 = None
        view_156: "f32[512]" = torch.ops.aten.view.default(sum_71, _shape_param_155);  sum_71 = _shape_param_155 = None
        convert_element_type_121: "bf16[512]" = torch.ops.prims.convert_element_type.default(view_156, torch.bfloat16);  view_156 = None
        view_157: "bf16[512, 49, 512]" = torch.ops.aten.view.default(mm_39, _shape_param_156);  mm_39 = _shape_param_156 = None
        convert_element_type_122: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_40, torch.float32);  mm_40 = None
        convert_element_type_123: "f32[512]" = torch.ops.prims.convert_element_type.default(convert_element_type_121, torch.float32);  convert_element_type_121 = None
        view_158: "bf16[512, 49, 16, 32]" = torch.ops.aten.view.default(view_157, _shape_param_157);  view_157 = _shape_param_157 = None
        permute_50: "bf16[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_158, [0, 2, 1, 3]);  view_158 = None
        clone_18: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(permute_50, memory_format = torch.contiguous_format);  permute_50 = None
        view_159: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_18, _shape_param_158);  clone_18 = _shape_param_158 = None
        bmm_16: "bf16[8192, 49, 32]" = torch.ops.aten.bmm.default(arg139_1, view_159);  arg139_1 = None
        bmm_17: "bf16[8192, 49, 49]" = torch.ops.aten.bmm.default(view_159, arg140_1);  view_159 = arg140_1 = None
        view_160: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(bmm_16, _shape_param_159);  bmm_16 = _shape_param_159 = None
        view_161: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(bmm_17, _shape_param_160);  bmm_17 = _shape_param_160 = None
        convert_element_type_124: "f32[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(view_161, torch.float32);  view_161 = None
        slice_5: "bf16[8192, 49, 56]" = torch.ops.aten.slice.Tensor(arg141_1, 1, 0, -7);  arg141_1 = None
        slice_6: "bf16[8192, 49, 49]" = torch.ops.aten.slice.Tensor(slice_5, 2, 0, -7);  slice_5 = None
        view_162: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(slice_6, _shape_param_161);  slice_6 = _shape_param_161 = None
        view_163: "i64[2401]" = torch.ops.aten.view.default(arg142_1, [-1]);  arg142_1 = None
        index_10: "f32[2401, 16]" = torch.ops.aten.index.Tensor(arg143_1, [view_163]);  arg143_1 = None
        view_164: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_10, _shape_param_162);  index_10 = _shape_param_162 = None
        permute_51: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_164, [2, 0, 1]);  view_164 = None
        clone_19: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
        unsqueeze_8: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_19, 0);  clone_19 = None
        add_25: "f32[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_162, unsqueeze_8);  view_162 = unsqueeze_8 = None
        view_165: "f32[128, 4, 16, 49, 49]" = torch.ops.aten.view.default(add_25, _shape_param_163);  add_25 = _shape_param_163 = None
        unsqueeze_9: "f32[4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(arg144_1, 1);  arg144_1 = None
        unsqueeze_10: "f32[1, 4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_9, 0);  unsqueeze_9 = None
        add_26: "f32[128, 4, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_165, unsqueeze_10);  view_165 = unsqueeze_10 = None
        view_166: "f32[512, 16, 49, 49]" = torch.ops.aten.view.default(add_26, _shape_param_164);  add_26 = _shape_param_164 = None
        sub_37: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(view_166, arg145_1);  view_166 = arg145_1 = None
        exp_9: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_37);  sub_37 = None
        div_26: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_9, arg146_1);  exp_9 = arg146_1 = None
        mul_130: "f32[512, 16, 49, 49]" = torch.ops.aten.mul.Tensor(convert_element_type_124, div_26);  convert_element_type_124 = None
        sum_72: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_130, [-1], True)
        neg_4: "f32[512, 16, 49, 49]" = torch.ops.aten.neg.default(div_26);  div_26 = None
        fma_4: "f32[512, 16, 49, 49]" = torch.ops.prims.fma.default(neg_4, sum_72, mul_130);  neg_4 = sum_72 = mul_130 = None
        convert_element_type_125: "bf16[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(fma_4, torch.bfloat16)
        sum_73: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_4, [0], True, dtype = torch.float32);  fma_4 = None
        squeeze_4: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_73, 0);  sum_73 = None
        permute_52: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_4, [1, 2, 0]);  squeeze_4 = None
        view_167: "f32[2401, 16]" = torch.ops.aten.view.default(permute_52, _shape_param_165);  permute_52 = _shape_param_165 = None
        index_put_4: "f32[169, 16]" = torch.ops.aten.index_put.default(full_1, [view_163], view_167, True);  view_163 = view_167 = None
        view_168: "bf16[8192, 49, 49]" = torch.ops.aten.view.default(convert_element_type_125, _shape_param_166);  convert_element_type_125 = _shape_param_166 = None
        bmm_18: "bf16[8192, 32, 49]" = torch.ops.aten.bmm.default(arg147_1, view_168);  arg147_1 = None
        bmm_19: "bf16[8192, 49, 32]" = torch.ops.aten.bmm.default(view_168, arg148_1);  view_168 = arg148_1 = None
        view_169: "bf16[512, 16, 32, 49]" = torch.ops.aten.view.default(bmm_18, _shape_param_167);  bmm_18 = _shape_param_167 = None
        view_170: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(bmm_19, _shape_param_168);  bmm_19 = _shape_param_168 = None
        permute_53: "bf16[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_169, [0, 1, 3, 2]);  view_169 = None
        mul_131: "bf16[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_170, 0.1767766952966369);  view_170 = None
        cat_4: "bf16[1536, 16, 49, 32]" = torch.ops.aten.cat.default([mul_131, permute_53, view_160]);  mul_131 = permute_53 = view_160 = None
        view_171: "bf16[3, 512, 16, 49, 32]" = torch.ops.aten.view.default(cat_4, _shape_param_169);  cat_4 = _shape_param_169 = None
        permute_54: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.permute.default(view_171, [1, 3, 0, 2, 4]);  view_171 = None
        clone_20: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_54, memory_format = torch.contiguous_format);  permute_54 = None
        view_172: "bf16[512, 49, 1536]" = torch.ops.aten.view.default(clone_20, _shape_param_170);  clone_20 = _shape_param_170 = None
        view_173: "bf16[25088, 1536]" = torch.ops.aten.view.default(view_172, _shape_param_171);  view_172 = _shape_param_171 = None
        mm_41: "bf16[25088, 512]" = torch.ops.aten.mm.default(view_173, arg149_1);  arg149_1 = None
        permute_55: "bf16[1536, 25088]" = torch.ops.aten.permute.default(view_173, [1, 0])
        mm_42: "bf16[1536, 512]" = torch.ops.aten.mm.default(permute_55, arg150_1);  permute_55 = arg150_1 = None
        sum_74: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_173, [0], True, dtype = torch.float32);  view_173 = None
        view_174: "f32[1536]" = torch.ops.aten.view.default(sum_74, _shape_param_172);  sum_74 = _shape_param_172 = None
        convert_element_type_126: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_174, torch.bfloat16);  view_174 = None
        view_175: "bf16[512, 49, 512]" = torch.ops.aten.view.default(mm_41, _shape_param_173);  mm_41 = _shape_param_173 = None
        convert_element_type_127: "f32[512, 49, 512]" = torch.ops.prims.convert_element_type.default(view_175, torch.float32);  view_175 = None
        convert_element_type_128: "f32[1536, 512]" = torch.ops.prims.convert_element_type.default(mm_42, torch.float32);  mm_42 = None
        convert_element_type_129: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_126, torch.float32);  convert_element_type_126 = None
        view_176: "f32[512, 7, 7, 512]" = torch.ops.aten.view.default(convert_element_type_127, _shape_param_174);  convert_element_type_127 = _shape_param_174 = None
        view_177: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_176, _shape_param_175);  view_176 = _shape_param_175 = None
        permute_56: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_177, [0, 1, 3, 2, 4, 5]);  view_177 = None
        clone_21: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_56, memory_format = torch.contiguous_format);  permute_56 = None
        view_178: "f32[128, 14, 14, 512]" = torch.ops.aten.view.default(clone_21, _shape_param_176);  clone_21 = _shape_param_176 = None
        index_11: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(view_178, [None, None, arg93_1]);  view_178 = None
        index_12: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_11, [None, arg93_1]);  index_11 = None
        mul_132: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(index_12, arg151_1);  arg151_1 = None
        mul_133: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_132, 512)
        sum_75: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_132, [3], True)
        convert_element_type_130: "f32[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(arg152_1, torch.float32);  arg152_1 = None
        sub_38: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_130, arg153_1);  convert_element_type_130 = arg153_1 = None
        mul_134: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_38, arg154_1);  sub_38 = None
        mul_135: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_132, mul_134);  mul_132 = None
        sum_76: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_135, [3], True);  mul_135 = None
        mul_136: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_134, sum_76);  sum_76 = None
        sub_39: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_133, sum_75);  mul_133 = sum_75 = None
        sub_40: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_39, mul_136);  sub_39 = mul_136 = None
        div_27: "f32[128, 14, 14, 1]" = torch.ops.aten.div.Tensor(arg154_1, 512);  arg154_1 = None
        mul_137: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_27, sub_40);  div_27 = sub_40 = None
        mul_138: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(index_12, mul_134);  mul_134 = None
        sum_77: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_138, [0, 1, 2]);  mul_138 = None
        sum_78: "f32[512]" = torch.ops.aten.sum.dim_IntList(index_12, [0, 1, 2]);  index_12 = None
        convert_element_type_131: "bf16[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(mul_137, torch.bfloat16);  mul_137 = None
        add_27: "bf16[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_151, convert_element_type_131);  view_151 = convert_element_type_131 = None
        view_179: "bf16[128, 196, 512]" = torch.ops.aten.view.default(add_27, _shape_param_177);  add_27 = _shape_param_177 = None
        convert_element_type_132: "bf16[128, 1, 1]" = torch.ops.prims.convert_element_type.default(arg155_1, torch.bfloat16);  arg155_1 = None
        div_28: "bf16[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_132, 0.9217391312122345);  convert_element_type_132 = None
        mul_139: "bf16[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_179, div_28);  div_28 = None
        view_180: "bf16[25088, 512]" = torch.ops.aten.view.default(mul_139, _shape_param_178);  mul_139 = _shape_param_178 = None
        mm_43: "bf16[25088, 2048]" = torch.ops.aten.mm.default(view_180, arg156_1);  arg156_1 = None
        permute_57: "bf16[512, 25088]" = torch.ops.aten.permute.default(view_180, [1, 0])
        mm_44: "bf16[512, 2048]" = torch.ops.aten.mm.default(permute_57, arg157_1);  permute_57 = arg157_1 = None
        sum_79: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_180, [0], True, dtype = torch.float32);  view_180 = None
        view_181: "f32[512]" = torch.ops.aten.view.default(sum_79, _shape_param_179);  sum_79 = _shape_param_179 = None
        convert_element_type_133: "bf16[512]" = torch.ops.prims.convert_element_type.default(view_181, torch.bfloat16);  view_181 = None
        view_182: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(mm_43, _shape_param_180);  mm_43 = _shape_param_180 = None
        convert_element_type_134: "f32[512, 2048]" = torch.ops.prims.convert_element_type.default(mm_44, torch.float32);  mm_44 = None
        convert_element_type_135: "f32[512]" = torch.ops.prims.convert_element_type.default(convert_element_type_133, torch.float32);  convert_element_type_133 = None
        convert_element_type_136: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_182, torch.float32);  view_182 = None
        view_183: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(arg158_1, _shape_param_181);  arg158_1 = _shape_param_181 = None
        convert_element_type_137: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_183, torch.float32);  view_183 = None
        mul_140: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_137, 0.7071067811865476)
        erf_5: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_140);  mul_140 = None
        add_28: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_141: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(add_28, 0.5);  add_28 = None
        mul_142: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_137, convert_element_type_137)
        mul_143: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_142, -0.5);  mul_142 = None
        exp_10: "f32[128, 196, 2048]" = torch.ops.aten.exp.default(mul_143);  mul_143 = None
        mul_144: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_10, 0.3989422804014327);  exp_10 = None
        mul_145: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_137, mul_144);  convert_element_type_137 = mul_144 = None
        add_29: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(mul_141, mul_145);  mul_141 = mul_145 = None
        mul_146: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_136, add_29);  convert_element_type_136 = add_29 = None
        convert_element_type_138: "bf16[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(mul_146, torch.bfloat16);  mul_146 = None
        view_184: "bf16[25088, 2048]" = torch.ops.aten.view.default(convert_element_type_138, _shape_param_182);  convert_element_type_138 = _shape_param_182 = None
        mm_45: "bf16[25088, 512]" = torch.ops.aten.mm.default(view_184, arg159_1);  arg159_1 = None
        permute_58: "bf16[2048, 25088]" = torch.ops.aten.permute.default(view_184, [1, 0])
        mm_46: "bf16[2048, 512]" = torch.ops.aten.mm.default(permute_58, arg160_1);  permute_58 = arg160_1 = None
        sum_80: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_184, [0], True, dtype = torch.float32);  view_184 = None
        view_185: "f32[2048]" = torch.ops.aten.view.default(sum_80, _shape_param_183);  sum_80 = _shape_param_183 = None
        convert_element_type_139: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_185, torch.bfloat16);  view_185 = None
        view_186: "bf16[128, 196, 512]" = torch.ops.aten.view.default(mm_45, _shape_param_184);  mm_45 = _shape_param_184 = None
        convert_element_type_140: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(view_186, torch.float32);  view_186 = None
        convert_element_type_141: "f32[2048, 512]" = torch.ops.prims.convert_element_type.default(mm_46, torch.float32);  mm_46 = None
        convert_element_type_142: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_139, torch.float32);  convert_element_type_139 = None
        mul_147: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_140, arg161_1);  arg161_1 = None
        mul_148: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_147, 512)
        sum_81: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_147, [2], True)
        convert_element_type_143: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(arg162_1, torch.float32);  arg162_1 = None
        sub_41: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_143, arg163_1);  convert_element_type_143 = arg163_1 = None
        mul_149: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_41, arg164_1);  sub_41 = None
        mul_150: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_147, mul_149);  mul_147 = None
        sum_82: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_150, [2], True);  mul_150 = None
        mul_151: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_149, sum_82);  sum_82 = None
        sub_42: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(mul_148, sum_81);  mul_148 = sum_81 = None
        sub_43: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(sub_42, mul_151);  sub_42 = mul_151 = None
        div_29: "f32[128, 196, 1]" = torch.ops.aten.div.Tensor(arg164_1, 512);  arg164_1 = None
        mul_152: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(div_29, sub_43);  div_29 = sub_43 = None
        mul_153: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_140, mul_149);  mul_149 = None
        sum_83: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_153, [0, 1]);  mul_153 = None
        sum_84: "f32[512]" = torch.ops.aten.sum.dim_IntList(convert_element_type_140, [0, 1]);  convert_element_type_140 = None
        convert_element_type_144: "bf16[128, 196, 512]" = torch.ops.prims.convert_element_type.default(mul_152, torch.bfloat16);  mul_152 = None
        add_30: "bf16[128, 196, 512]" = torch.ops.aten.add.Tensor(view_179, convert_element_type_144);  view_179 = convert_element_type_144 = None
        view_187: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(add_30, _shape_param_185);  add_30 = _shape_param_185 = None
        convert_element_type_145: "bf16[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(arg165_1, torch.bfloat16);  arg165_1 = None
        div_30: "bf16[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_145, 0.9217391312122345);  convert_element_type_145 = None
        mul_154: "bf16[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_187, div_30);  div_30 = None
        view_188: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(mul_154, _shape_param_186);  mul_154 = _shape_param_186 = None
        permute_59: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_188, [0, 1, 3, 2, 4, 5]);  view_188 = None
        clone_22: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_59, memory_format = torch.contiguous_format);  permute_59 = None
        view_189: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(clone_22, _shape_param_187);  clone_22 = _shape_param_187 = None
        view_190: "bf16[512, 49, 512]" = torch.ops.aten.view.default(view_189, _shape_param_188);  view_189 = _shape_param_188 = None
        view_191: "bf16[25088, 512]" = torch.ops.aten.view.default(view_190, _shape_param_189);  view_190 = _shape_param_189 = None
        mm_47: "bf16[25088, 512]" = torch.ops.aten.mm.default(view_191, arg166_1);  arg166_1 = None
        permute_60: "bf16[512, 25088]" = torch.ops.aten.permute.default(view_191, [1, 0])
        mm_48: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_60, arg167_1);  permute_60 = arg167_1 = None
        sum_85: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_191, [0], True, dtype = torch.float32);  view_191 = None
        view_192: "f32[512]" = torch.ops.aten.view.default(sum_85, _shape_param_190);  sum_85 = _shape_param_190 = None
        convert_element_type_146: "bf16[512]" = torch.ops.prims.convert_element_type.default(view_192, torch.bfloat16);  view_192 = None
        view_193: "bf16[512, 49, 512]" = torch.ops.aten.view.default(mm_47, _shape_param_191);  mm_47 = _shape_param_191 = None
        convert_element_type_147: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_48, torch.float32);  mm_48 = None
        convert_element_type_148: "f32[512]" = torch.ops.prims.convert_element_type.default(convert_element_type_146, torch.float32);  convert_element_type_146 = None
        view_194: "bf16[512, 49, 16, 32]" = torch.ops.aten.view.default(view_193, _shape_param_192);  view_193 = _shape_param_192 = None
        permute_61: "bf16[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_194, [0, 2, 1, 3]);  view_194 = None
        clone_23: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(permute_61, memory_format = torch.contiguous_format);  permute_61 = None
        view_195: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_23, _shape_param_193);  clone_23 = _shape_param_193 = None
        bmm_20: "bf16[8192, 49, 32]" = torch.ops.aten.bmm.default(arg168_1, view_195);  arg168_1 = None
        bmm_21: "bf16[8192, 49, 49]" = torch.ops.aten.bmm.default(view_195, arg169_1);  view_195 = arg169_1 = None
        view_196: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(bmm_20, _shape_param_194);  bmm_20 = _shape_param_194 = None
        view_197: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(bmm_21, _shape_param_195);  bmm_21 = _shape_param_195 = None
        convert_element_type_149: "f32[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(view_197, torch.float32);  view_197 = None
        slice_7: "bf16[8192, 49, 56]" = torch.ops.aten.slice.Tensor(arg170_1, 1, 0, -7);  arg170_1 = None
        slice_8: "bf16[8192, 49, 49]" = torch.ops.aten.slice.Tensor(slice_7, 2, 0, -7);  slice_7 = None
        view_198: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(slice_8, _shape_param_196);  slice_8 = _shape_param_196 = None
        view_199: "i64[2401]" = torch.ops.aten.view.default(arg171_1, [-1]);  arg171_1 = None
        index_13: "f32[2401, 16]" = torch.ops.aten.index.Tensor(arg172_1, [view_199]);  arg172_1 = None
        view_200: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_13, _shape_param_197);  index_13 = _shape_param_197 = None
        permute_62: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_200, [2, 0, 1]);  view_200 = None
        clone_24: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
        unsqueeze_11: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_24, 0);  clone_24 = None
        add_31: "f32[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_198, unsqueeze_11);  view_198 = unsqueeze_11 = None
        sub_44: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(add_31, arg173_1);  add_31 = arg173_1 = None
        exp_11: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_44);  sub_44 = None
        div_31: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_11, arg174_1);  exp_11 = arg174_1 = None
        mul_155: "f32[512, 16, 49, 49]" = torch.ops.aten.mul.Tensor(convert_element_type_149, div_31);  convert_element_type_149 = None
        sum_86: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_155, [-1], True)
        neg_5: "f32[512, 16, 49, 49]" = torch.ops.aten.neg.default(div_31);  div_31 = None
        fma_5: "f32[512, 16, 49, 49]" = torch.ops.prims.fma.default(neg_5, sum_86, mul_155);  neg_5 = sum_86 = mul_155 = None
        convert_element_type_150: "bf16[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(fma_5, torch.bfloat16)
        sum_87: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_5, [0], True, dtype = torch.float32);  fma_5 = None
        squeeze_5: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_87, 0);  sum_87 = None
        permute_63: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_5, [1, 2, 0]);  squeeze_5 = None
        view_201: "f32[2401, 16]" = torch.ops.aten.view.default(permute_63, _shape_param_198);  permute_63 = _shape_param_198 = None
        index_put_5: "f32[169, 16]" = torch.ops.aten.index_put.default(full_1, [view_199], view_201, True);  view_199 = view_201 = None
        view_202: "bf16[8192, 49, 49]" = torch.ops.aten.view.default(convert_element_type_150, _shape_param_199);  convert_element_type_150 = _shape_param_199 = None
        bmm_22: "bf16[8192, 32, 49]" = torch.ops.aten.bmm.default(arg175_1, view_202);  arg175_1 = None
        bmm_23: "bf16[8192, 49, 32]" = torch.ops.aten.bmm.default(view_202, arg176_1);  view_202 = arg176_1 = None
        view_203: "bf16[512, 16, 32, 49]" = torch.ops.aten.view.default(bmm_22, _shape_param_200);  bmm_22 = _shape_param_200 = None
        view_204: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(bmm_23, _shape_param_201);  bmm_23 = _shape_param_201 = None
        permute_64: "bf16[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_203, [0, 1, 3, 2]);  view_203 = None
        mul_156: "bf16[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_204, 0.1767766952966369);  view_204 = None
        cat_5: "bf16[1536, 16, 49, 32]" = torch.ops.aten.cat.default([mul_156, permute_64, view_196]);  mul_156 = permute_64 = view_196 = None
        view_205: "bf16[3, 512, 16, 49, 32]" = torch.ops.aten.view.default(cat_5, _shape_param_202);  cat_5 = _shape_param_202 = None
        permute_65: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.permute.default(view_205, [1, 3, 0, 2, 4]);  view_205 = None
        clone_25: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_65, memory_format = torch.contiguous_format);  permute_65 = None
        view_206: "bf16[512, 49, 1536]" = torch.ops.aten.view.default(clone_25, _shape_param_203);  clone_25 = _shape_param_203 = None
        view_207: "bf16[25088, 1536]" = torch.ops.aten.view.default(view_206, _shape_param_204);  view_206 = _shape_param_204 = None
        mm_49: "bf16[25088, 512]" = torch.ops.aten.mm.default(view_207, arg177_1);  arg177_1 = None
        permute_66: "bf16[1536, 25088]" = torch.ops.aten.permute.default(view_207, [1, 0])
        mm_50: "bf16[1536, 512]" = torch.ops.aten.mm.default(permute_66, arg178_1);  permute_66 = arg178_1 = None
        sum_88: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_207, [0], True, dtype = torch.float32);  view_207 = None
        view_208: "f32[1536]" = torch.ops.aten.view.default(sum_88, _shape_param_205);  sum_88 = _shape_param_205 = None
        convert_element_type_151: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_208, torch.bfloat16);  view_208 = None
        view_209: "bf16[512, 49, 512]" = torch.ops.aten.view.default(mm_49, _shape_param_206);  mm_49 = _shape_param_206 = None
        convert_element_type_152: "f32[512, 49, 512]" = torch.ops.prims.convert_element_type.default(view_209, torch.float32);  view_209 = None
        convert_element_type_153: "f32[1536, 512]" = torch.ops.prims.convert_element_type.default(mm_50, torch.float32);  mm_50 = None
        convert_element_type_154: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_151, torch.float32);  convert_element_type_151 = None
        view_210: "f32[512, 7, 7, 512]" = torch.ops.aten.view.default(convert_element_type_152, _shape_param_207);  convert_element_type_152 = _shape_param_207 = None
        view_211: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_210, _shape_param_208);  view_210 = _shape_param_208 = None
        permute_67: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_211, [0, 1, 3, 2, 4, 5]);  view_211 = None
        clone_26: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_67, memory_format = torch.contiguous_format);  permute_67 = None
        view_212: "f32[128, 14, 14, 512]" = torch.ops.aten.view.default(clone_26, _shape_param_209);  clone_26 = _shape_param_209 = None
        mul_157: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_212, arg179_1);  arg179_1 = None
        mul_158: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_157, 512)
        sum_89: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_157, [3], True)
        convert_element_type_155: "f32[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(arg180_1, torch.float32);  arg180_1 = None
        sub_45: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_155, arg181_1);  convert_element_type_155 = arg181_1 = None
        mul_159: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_45, arg182_1);  sub_45 = None
        mul_160: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_157, mul_159);  mul_157 = None
        sum_90: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_160, [3], True);  mul_160 = None
        mul_161: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_159, sum_90);  sum_90 = None
        sub_46: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_158, sum_89);  mul_158 = sum_89 = None
        sub_47: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_46, mul_161);  sub_46 = mul_161 = None
        div_32: "f32[128, 14, 14, 1]" = torch.ops.aten.div.Tensor(arg182_1, 512);  arg182_1 = None
        mul_162: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_32, sub_47);  div_32 = sub_47 = None
        mul_163: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_212, mul_159);  mul_159 = None
        sum_91: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_163, [0, 1, 2]);  mul_163 = None
        sum_92: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_212, [0, 1, 2]);  view_212 = None
        convert_element_type_156: "bf16[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(mul_162, torch.bfloat16);  mul_162 = None
        add_32: "bf16[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_187, convert_element_type_156);  view_187 = convert_element_type_156 = None
        view_213: "bf16[128, 196, 512]" = torch.ops.aten.view.default(add_32, _shape_param_210);  add_32 = _shape_param_210 = None
        convert_element_type_157: "bf16[128, 1, 1]" = torch.ops.prims.convert_element_type.default(arg183_1, torch.bfloat16);  arg183_1 = None
        div_33: "bf16[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_157, 0.9260869547724724);  convert_element_type_157 = None
        mul_164: "bf16[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_213, div_33);  div_33 = None
        view_214: "bf16[25088, 512]" = torch.ops.aten.view.default(mul_164, _shape_param_211);  mul_164 = _shape_param_211 = None
        mm_51: "bf16[25088, 2048]" = torch.ops.aten.mm.default(view_214, arg184_1);  arg184_1 = None
        permute_68: "bf16[512, 25088]" = torch.ops.aten.permute.default(view_214, [1, 0])
        mm_52: "bf16[512, 2048]" = torch.ops.aten.mm.default(permute_68, arg185_1);  permute_68 = arg185_1 = None
        sum_93: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_214, [0], True, dtype = torch.float32);  view_214 = None
        view_215: "f32[512]" = torch.ops.aten.view.default(sum_93, _shape_param_212);  sum_93 = _shape_param_212 = None
        convert_element_type_158: "bf16[512]" = torch.ops.prims.convert_element_type.default(view_215, torch.bfloat16);  view_215 = None
        view_216: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(mm_51, _shape_param_213);  mm_51 = _shape_param_213 = None
        convert_element_type_159: "f32[512, 2048]" = torch.ops.prims.convert_element_type.default(mm_52, torch.float32);  mm_52 = None
        convert_element_type_160: "f32[512]" = torch.ops.prims.convert_element_type.default(convert_element_type_158, torch.float32);  convert_element_type_158 = None
        convert_element_type_161: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_216, torch.float32);  view_216 = None
        view_217: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(arg186_1, _shape_param_214);  arg186_1 = _shape_param_214 = None
        convert_element_type_162: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_217, torch.float32);  view_217 = None
        mul_165: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_162, 0.7071067811865476)
        erf_6: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_165);  mul_165 = None
        add_33: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_166: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(add_33, 0.5);  add_33 = None
        mul_167: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_162, convert_element_type_162)
        mul_168: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_167, -0.5);  mul_167 = None
        exp_12: "f32[128, 196, 2048]" = torch.ops.aten.exp.default(mul_168);  mul_168 = None
        mul_169: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_12, 0.3989422804014327);  exp_12 = None
        mul_170: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_162, mul_169);  convert_element_type_162 = mul_169 = None
        add_34: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(mul_166, mul_170);  mul_166 = mul_170 = None
        mul_171: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_161, add_34);  convert_element_type_161 = add_34 = None
        convert_element_type_163: "bf16[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(mul_171, torch.bfloat16);  mul_171 = None
        view_218: "bf16[25088, 2048]" = torch.ops.aten.view.default(convert_element_type_163, _shape_param_215);  convert_element_type_163 = _shape_param_215 = None
        mm_53: "bf16[25088, 512]" = torch.ops.aten.mm.default(view_218, arg187_1);  arg187_1 = None
        permute_69: "bf16[2048, 25088]" = torch.ops.aten.permute.default(view_218, [1, 0])
        mm_54: "bf16[2048, 512]" = torch.ops.aten.mm.default(permute_69, arg188_1);  permute_69 = arg188_1 = None
        sum_94: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_218, [0], True, dtype = torch.float32);  view_218 = None
        view_219: "f32[2048]" = torch.ops.aten.view.default(sum_94, _shape_param_216);  sum_94 = _shape_param_216 = None
        convert_element_type_164: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_219, torch.bfloat16);  view_219 = None
        view_220: "bf16[128, 196, 512]" = torch.ops.aten.view.default(mm_53, _shape_param_217);  mm_53 = _shape_param_217 = None
        convert_element_type_165: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(view_220, torch.float32);  view_220 = None
        convert_element_type_166: "f32[2048, 512]" = torch.ops.prims.convert_element_type.default(mm_54, torch.float32);  mm_54 = None
        convert_element_type_167: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_164, torch.float32);  convert_element_type_164 = None
        mul_172: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_165, arg189_1);  arg189_1 = None
        mul_173: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_172, 512)
        sum_95: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_172, [2], True)
        convert_element_type_168: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(arg190_1, torch.float32);  arg190_1 = None
        sub_48: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_168, arg191_1);  convert_element_type_168 = arg191_1 = None
        mul_174: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_48, arg192_1);  sub_48 = None
        mul_175: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_172, mul_174);  mul_172 = None
        sum_96: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_175, [2], True);  mul_175 = None
        mul_176: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_174, sum_96);  sum_96 = None
        sub_49: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(mul_173, sum_95);  mul_173 = sum_95 = None
        sub_50: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(sub_49, mul_176);  sub_49 = mul_176 = None
        div_34: "f32[128, 196, 1]" = torch.ops.aten.div.Tensor(arg192_1, 512);  arg192_1 = None
        mul_177: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(div_34, sub_50);  div_34 = sub_50 = None
        mul_178: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_165, mul_174);  mul_174 = None
        sum_97: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_178, [0, 1]);  mul_178 = None
        sum_98: "f32[512]" = torch.ops.aten.sum.dim_IntList(convert_element_type_165, [0, 1]);  convert_element_type_165 = None
        convert_element_type_169: "bf16[128, 196, 512]" = torch.ops.prims.convert_element_type.default(mul_177, torch.bfloat16);  mul_177 = None
        add_35: "bf16[128, 196, 512]" = torch.ops.aten.add.Tensor(view_213, convert_element_type_169);  view_213 = convert_element_type_169 = None
        view_221: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(add_35, _shape_param_218);  add_35 = _shape_param_218 = None
        convert_element_type_170: "bf16[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(arg193_1, torch.bfloat16);  arg193_1 = None
        div_35: "bf16[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_170, 0.9260869547724724);  convert_element_type_170 = None
        mul_179: "bf16[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_221, div_35);  div_35 = None
        index_14: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(mul_179, [None, None, fmod]);  mul_179 = None
        index_15: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_14, [None, fmod]);  index_14 = None
        view_222: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(index_15, _shape_param_219);  index_15 = _shape_param_219 = None
        permute_70: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_222, [0, 1, 3, 2, 4, 5]);  view_222 = None
        clone_27: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_70, memory_format = torch.contiguous_format);  permute_70 = None
        view_223: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(clone_27, _shape_param_220);  clone_27 = _shape_param_220 = None
        view_224: "bf16[512, 49, 512]" = torch.ops.aten.view.default(view_223, _shape_param_221);  view_223 = _shape_param_221 = None
        view_225: "bf16[25088, 512]" = torch.ops.aten.view.default(view_224, _shape_param_222);  view_224 = _shape_param_222 = None
        mm_55: "bf16[25088, 512]" = torch.ops.aten.mm.default(view_225, arg194_1);  arg194_1 = None
        permute_71: "bf16[512, 25088]" = torch.ops.aten.permute.default(view_225, [1, 0])
        mm_56: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_71, arg195_1);  permute_71 = arg195_1 = None
        sum_99: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_225, [0], True, dtype = torch.float32);  view_225 = None
        view_226: "f32[512]" = torch.ops.aten.view.default(sum_99, _shape_param_223);  sum_99 = _shape_param_223 = None
        convert_element_type_171: "bf16[512]" = torch.ops.prims.convert_element_type.default(view_226, torch.bfloat16);  view_226 = None
        view_227: "bf16[512, 49, 512]" = torch.ops.aten.view.default(mm_55, _shape_param_224);  mm_55 = _shape_param_224 = None
        convert_element_type_172: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_56, torch.float32);  mm_56 = None
        convert_element_type_173: "f32[512]" = torch.ops.prims.convert_element_type.default(convert_element_type_171, torch.float32);  convert_element_type_171 = None
        view_228: "bf16[512, 49, 16, 32]" = torch.ops.aten.view.default(view_227, _shape_param_225);  view_227 = _shape_param_225 = None
        permute_72: "bf16[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_228, [0, 2, 1, 3]);  view_228 = None
        clone_28: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(permute_72, memory_format = torch.contiguous_format);  permute_72 = None
        view_229: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_28, _shape_param_226);  clone_28 = _shape_param_226 = None
        bmm_24: "bf16[8192, 49, 32]" = torch.ops.aten.bmm.default(arg196_1, view_229);  arg196_1 = None
        bmm_25: "bf16[8192, 49, 49]" = torch.ops.aten.bmm.default(view_229, arg197_1);  view_229 = arg197_1 = None
        view_230: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(bmm_24, _shape_param_227);  bmm_24 = _shape_param_227 = None
        view_231: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(bmm_25, _shape_param_228);  bmm_25 = _shape_param_228 = None
        convert_element_type_174: "f32[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(view_231, torch.float32);  view_231 = None
        slice_9: "bf16[8192, 49, 56]" = torch.ops.aten.slice.Tensor(arg198_1, 1, 0, -7);  arg198_1 = None
        slice_10: "bf16[8192, 49, 49]" = torch.ops.aten.slice.Tensor(slice_9, 2, 0, -7);  slice_9 = None
        view_232: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(slice_10, _shape_param_229);  slice_10 = _shape_param_229 = None
        view_233: "i64[2401]" = torch.ops.aten.view.default(arg199_1, [-1]);  arg199_1 = None
        index_16: "f32[2401, 16]" = torch.ops.aten.index.Tensor(arg200_1, [view_233]);  arg200_1 = None
        view_234: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_16, _shape_param_230);  index_16 = _shape_param_230 = None
        permute_73: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_234, [2, 0, 1]);  view_234 = None
        clone_29: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
        unsqueeze_12: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_29, 0);  clone_29 = None
        add_36: "f32[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_232, unsqueeze_12);  view_232 = unsqueeze_12 = None
        view_235: "f32[128, 4, 16, 49, 49]" = torch.ops.aten.view.default(add_36, _shape_param_231);  add_36 = _shape_param_231 = None
        unsqueeze_13: "f32[4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(arg201_1, 1);  arg201_1 = None
        unsqueeze_14: "f32[1, 4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_13, 0);  unsqueeze_13 = None
        add_37: "f32[128, 4, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_235, unsqueeze_14);  view_235 = unsqueeze_14 = None
        view_236: "f32[512, 16, 49, 49]" = torch.ops.aten.view.default(add_37, _shape_param_232);  add_37 = _shape_param_232 = None
        sub_51: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(view_236, arg202_1);  view_236 = arg202_1 = None
        exp_13: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_51);  sub_51 = None
        div_36: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_13, arg203_1);  exp_13 = arg203_1 = None
        mul_180: "f32[512, 16, 49, 49]" = torch.ops.aten.mul.Tensor(convert_element_type_174, div_36);  convert_element_type_174 = None
        sum_100: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_180, [-1], True)
        neg_6: "f32[512, 16, 49, 49]" = torch.ops.aten.neg.default(div_36);  div_36 = None
        fma_6: "f32[512, 16, 49, 49]" = torch.ops.prims.fma.default(neg_6, sum_100, mul_180);  neg_6 = sum_100 = mul_180 = None
        convert_element_type_175: "bf16[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(fma_6, torch.bfloat16)
        sum_101: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_6, [0], True, dtype = torch.float32);  fma_6 = None
        squeeze_6: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_101, 0);  sum_101 = None
        permute_74: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_6, [1, 2, 0]);  squeeze_6 = None
        view_237: "f32[2401, 16]" = torch.ops.aten.view.default(permute_74, _shape_param_233);  permute_74 = _shape_param_233 = None
        index_put_6: "f32[169, 16]" = torch.ops.aten.index_put.default(full_1, [view_233], view_237, True);  view_233 = view_237 = None
        view_238: "bf16[8192, 49, 49]" = torch.ops.aten.view.default(convert_element_type_175, _shape_param_234);  convert_element_type_175 = _shape_param_234 = None
        bmm_26: "bf16[8192, 32, 49]" = torch.ops.aten.bmm.default(arg204_1, view_238);  arg204_1 = None
        bmm_27: "bf16[8192, 49, 32]" = torch.ops.aten.bmm.default(view_238, arg205_1);  view_238 = arg205_1 = None
        view_239: "bf16[512, 16, 32, 49]" = torch.ops.aten.view.default(bmm_26, _shape_param_235);  bmm_26 = _shape_param_235 = None
        view_240: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(bmm_27, _shape_param_236);  bmm_27 = _shape_param_236 = None
        permute_75: "bf16[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_239, [0, 1, 3, 2]);  view_239 = None
        mul_181: "bf16[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_240, 0.1767766952966369);  view_240 = None
        cat_6: "bf16[1536, 16, 49, 32]" = torch.ops.aten.cat.default([mul_181, permute_75, view_230]);  mul_181 = permute_75 = view_230 = None
        view_241: "bf16[3, 512, 16, 49, 32]" = torch.ops.aten.view.default(cat_6, _shape_param_237);  cat_6 = _shape_param_237 = None
        permute_76: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.permute.default(view_241, [1, 3, 0, 2, 4]);  view_241 = None
        clone_30: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_76, memory_format = torch.contiguous_format);  permute_76 = None
        view_242: "bf16[512, 49, 1536]" = torch.ops.aten.view.default(clone_30, _shape_param_238);  clone_30 = _shape_param_238 = None
        view_243: "bf16[25088, 1536]" = torch.ops.aten.view.default(view_242, _shape_param_239);  view_242 = _shape_param_239 = None
        mm_57: "bf16[25088, 512]" = torch.ops.aten.mm.default(view_243, arg206_1);  arg206_1 = None
        permute_77: "bf16[1536, 25088]" = torch.ops.aten.permute.default(view_243, [1, 0])
        mm_58: "bf16[1536, 512]" = torch.ops.aten.mm.default(permute_77, arg207_1);  permute_77 = arg207_1 = None
        sum_102: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_243, [0], True, dtype = torch.float32);  view_243 = None
        view_244: "f32[1536]" = torch.ops.aten.view.default(sum_102, _shape_param_240);  sum_102 = _shape_param_240 = None
        convert_element_type_176: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_244, torch.bfloat16);  view_244 = None
        view_245: "bf16[512, 49, 512]" = torch.ops.aten.view.default(mm_57, _shape_param_241);  mm_57 = _shape_param_241 = None
        convert_element_type_177: "f32[512, 49, 512]" = torch.ops.prims.convert_element_type.default(view_245, torch.float32);  view_245 = None
        convert_element_type_178: "f32[1536, 512]" = torch.ops.prims.convert_element_type.default(mm_58, torch.float32);  mm_58 = None
        convert_element_type_179: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_176, torch.float32);  convert_element_type_176 = None
        view_246: "f32[512, 7, 7, 512]" = torch.ops.aten.view.default(convert_element_type_177, _shape_param_242);  convert_element_type_177 = _shape_param_242 = None
        view_247: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_246, _shape_param_243);  view_246 = _shape_param_243 = None
        permute_78: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_247, [0, 1, 3, 2, 4, 5]);  view_247 = None
        clone_31: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_78, memory_format = torch.contiguous_format);  permute_78 = None
        view_248: "f32[128, 14, 14, 512]" = torch.ops.aten.view.default(clone_31, _shape_param_244);  clone_31 = _shape_param_244 = None
        index_17: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(view_248, [None, None, arg93_1]);  view_248 = None
        index_18: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_17, [None, arg93_1]);  index_17 = None
        mul_182: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(index_18, arg208_1);  arg208_1 = None
        mul_183: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_182, 512)
        sum_103: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_182, [3], True)
        convert_element_type_180: "f32[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(arg209_1, torch.float32);  arg209_1 = None
        sub_52: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_180, arg210_1);  convert_element_type_180 = arg210_1 = None
        mul_184: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_52, arg211_1);  sub_52 = None
        mul_185: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_182, mul_184);  mul_182 = None
        sum_104: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_185, [3], True);  mul_185 = None
        mul_186: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_184, sum_104);  sum_104 = None
        sub_53: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_183, sum_103);  mul_183 = sum_103 = None
        sub_54: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_53, mul_186);  sub_53 = mul_186 = None
        div_37: "f32[128, 14, 14, 1]" = torch.ops.aten.div.Tensor(arg211_1, 512);  arg211_1 = None
        mul_187: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_37, sub_54);  div_37 = sub_54 = None
        mul_188: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(index_18, mul_184);  mul_184 = None
        sum_105: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_188, [0, 1, 2]);  mul_188 = None
        sum_106: "f32[512]" = torch.ops.aten.sum.dim_IntList(index_18, [0, 1, 2]);  index_18 = None
        convert_element_type_181: "bf16[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(mul_187, torch.bfloat16);  mul_187 = None
        add_38: "bf16[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_221, convert_element_type_181);  view_221 = convert_element_type_181 = None
        view_249: "bf16[128, 196, 512]" = torch.ops.aten.view.default(add_38, _shape_param_245);  add_38 = _shape_param_245 = None
        convert_element_type_182: "bf16[128, 1, 1]" = torch.ops.prims.convert_element_type.default(arg212_1, torch.bfloat16);  arg212_1 = None
        div_38: "bf16[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_182, 0.9304347857832909);  convert_element_type_182 = None
        mul_189: "bf16[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_249, div_38);  div_38 = None
        view_250: "bf16[25088, 512]" = torch.ops.aten.view.default(mul_189, _shape_param_246);  mul_189 = _shape_param_246 = None
        mm_59: "bf16[25088, 2048]" = torch.ops.aten.mm.default(view_250, arg213_1);  arg213_1 = None
        permute_79: "bf16[512, 25088]" = torch.ops.aten.permute.default(view_250, [1, 0])
        mm_60: "bf16[512, 2048]" = torch.ops.aten.mm.default(permute_79, arg214_1);  permute_79 = arg214_1 = None
        sum_107: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_250, [0], True, dtype = torch.float32);  view_250 = None
        view_251: "f32[512]" = torch.ops.aten.view.default(sum_107, _shape_param_247);  sum_107 = _shape_param_247 = None
        convert_element_type_183: "bf16[512]" = torch.ops.prims.convert_element_type.default(view_251, torch.bfloat16);  view_251 = None
        view_252: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(mm_59, _shape_param_248);  mm_59 = _shape_param_248 = None
        convert_element_type_184: "f32[512, 2048]" = torch.ops.prims.convert_element_type.default(mm_60, torch.float32);  mm_60 = None
        convert_element_type_185: "f32[512]" = torch.ops.prims.convert_element_type.default(convert_element_type_183, torch.float32);  convert_element_type_183 = None
        convert_element_type_186: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_252, torch.float32);  view_252 = None
        view_253: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(arg215_1, _shape_param_249);  arg215_1 = _shape_param_249 = None
        convert_element_type_187: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_253, torch.float32);  view_253 = None
        mul_190: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_187, 0.7071067811865476)
        erf_7: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_190);  mul_190 = None
        add_39: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_191: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(add_39, 0.5);  add_39 = None
        mul_192: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_187, convert_element_type_187)
        mul_193: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_192, -0.5);  mul_192 = None
        exp_14: "f32[128, 196, 2048]" = torch.ops.aten.exp.default(mul_193);  mul_193 = None
        mul_194: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_14, 0.3989422804014327);  exp_14 = None
        mul_195: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_187, mul_194);  convert_element_type_187 = mul_194 = None
        add_40: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(mul_191, mul_195);  mul_191 = mul_195 = None
        mul_196: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_186, add_40);  convert_element_type_186 = add_40 = None
        convert_element_type_188: "bf16[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(mul_196, torch.bfloat16);  mul_196 = None
        view_254: "bf16[25088, 2048]" = torch.ops.aten.view.default(convert_element_type_188, _shape_param_250);  convert_element_type_188 = _shape_param_250 = None
        mm_61: "bf16[25088, 512]" = torch.ops.aten.mm.default(view_254, arg216_1);  arg216_1 = None
        permute_80: "bf16[2048, 25088]" = torch.ops.aten.permute.default(view_254, [1, 0])
        mm_62: "bf16[2048, 512]" = torch.ops.aten.mm.default(permute_80, arg217_1);  permute_80 = arg217_1 = None
        sum_108: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_254, [0], True, dtype = torch.float32);  view_254 = None
        view_255: "f32[2048]" = torch.ops.aten.view.default(sum_108, _shape_param_251);  sum_108 = _shape_param_251 = None
        convert_element_type_189: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_255, torch.bfloat16);  view_255 = None
        view_256: "bf16[128, 196, 512]" = torch.ops.aten.view.default(mm_61, _shape_param_252);  mm_61 = _shape_param_252 = None
        convert_element_type_190: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(view_256, torch.float32);  view_256 = None
        convert_element_type_191: "f32[2048, 512]" = torch.ops.prims.convert_element_type.default(mm_62, torch.float32);  mm_62 = None
        convert_element_type_192: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_189, torch.float32);  convert_element_type_189 = None
        mul_197: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_190, arg218_1);  arg218_1 = None
        mul_198: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_197, 512)
        sum_109: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_197, [2], True)
        convert_element_type_193: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(arg219_1, torch.float32);  arg219_1 = None
        sub_55: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_193, arg220_1);  convert_element_type_193 = arg220_1 = None
        mul_199: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_55, arg221_1);  sub_55 = None
        mul_200: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_197, mul_199);  mul_197 = None
        sum_110: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_200, [2], True);  mul_200 = None
        mul_201: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_199, sum_110);  sum_110 = None
        sub_56: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(mul_198, sum_109);  mul_198 = sum_109 = None
        sub_57: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(sub_56, mul_201);  sub_56 = mul_201 = None
        div_39: "f32[128, 196, 1]" = torch.ops.aten.div.Tensor(arg221_1, 512);  arg221_1 = None
        mul_202: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(div_39, sub_57);  div_39 = sub_57 = None
        mul_203: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_190, mul_199);  mul_199 = None
        sum_111: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_203, [0, 1]);  mul_203 = None
        sum_112: "f32[512]" = torch.ops.aten.sum.dim_IntList(convert_element_type_190, [0, 1]);  convert_element_type_190 = None
        convert_element_type_194: "bf16[128, 196, 512]" = torch.ops.prims.convert_element_type.default(mul_202, torch.bfloat16);  mul_202 = None
        add_41: "bf16[128, 196, 512]" = torch.ops.aten.add.Tensor(view_249, convert_element_type_194);  view_249 = convert_element_type_194 = None
        view_257: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(add_41, _shape_param_253);  add_41 = _shape_param_253 = None
        convert_element_type_195: "bf16[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(arg222_1, torch.bfloat16);  arg222_1 = None
        div_40: "bf16[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_195, 0.9304347857832909);  convert_element_type_195 = None
        mul_204: "bf16[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_257, div_40);  div_40 = None
        view_258: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(mul_204, _shape_param_254);  mul_204 = _shape_param_254 = None
        permute_81: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_258, [0, 1, 3, 2, 4, 5]);  view_258 = None
        clone_32: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_81, memory_format = torch.contiguous_format);  permute_81 = None
        view_259: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(clone_32, _shape_param_255);  clone_32 = _shape_param_255 = None
        view_260: "bf16[512, 49, 512]" = torch.ops.aten.view.default(view_259, _shape_param_256);  view_259 = _shape_param_256 = None
        view_261: "bf16[25088, 512]" = torch.ops.aten.view.default(view_260, _shape_param_257);  view_260 = _shape_param_257 = None
        mm_63: "bf16[25088, 512]" = torch.ops.aten.mm.default(view_261, arg223_1);  arg223_1 = None
        permute_82: "bf16[512, 25088]" = torch.ops.aten.permute.default(view_261, [1, 0])
        mm_64: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_82, arg224_1);  permute_82 = arg224_1 = None
        sum_113: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_261, [0], True, dtype = torch.float32);  view_261 = None
        view_262: "f32[512]" = torch.ops.aten.view.default(sum_113, _shape_param_258);  sum_113 = _shape_param_258 = None
        convert_element_type_196: "bf16[512]" = torch.ops.prims.convert_element_type.default(view_262, torch.bfloat16);  view_262 = None
        view_263: "bf16[512, 49, 512]" = torch.ops.aten.view.default(mm_63, _shape_param_259);  mm_63 = _shape_param_259 = None
        convert_element_type_197: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_64, torch.float32);  mm_64 = None
        convert_element_type_198: "f32[512]" = torch.ops.prims.convert_element_type.default(convert_element_type_196, torch.float32);  convert_element_type_196 = None
        view_264: "bf16[512, 49, 16, 32]" = torch.ops.aten.view.default(view_263, _shape_param_260);  view_263 = _shape_param_260 = None
        permute_83: "bf16[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_264, [0, 2, 1, 3]);  view_264 = None
        clone_33: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(permute_83, memory_format = torch.contiguous_format);  permute_83 = None
        view_265: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_33, _shape_param_261);  clone_33 = _shape_param_261 = None
        bmm_28: "bf16[8192, 49, 32]" = torch.ops.aten.bmm.default(arg225_1, view_265);  arg225_1 = None
        bmm_29: "bf16[8192, 49, 49]" = torch.ops.aten.bmm.default(view_265, arg226_1);  view_265 = arg226_1 = None
        view_266: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(bmm_28, _shape_param_262);  bmm_28 = _shape_param_262 = None
        view_267: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(bmm_29, _shape_param_263);  bmm_29 = _shape_param_263 = None
        convert_element_type_199: "f32[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(view_267, torch.float32);  view_267 = None
        slice_11: "bf16[8192, 49, 56]" = torch.ops.aten.slice.Tensor(arg227_1, 1, 0, -7);  arg227_1 = None
        slice_12: "bf16[8192, 49, 49]" = torch.ops.aten.slice.Tensor(slice_11, 2, 0, -7);  slice_11 = None
        view_268: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(slice_12, _shape_param_264);  slice_12 = _shape_param_264 = None
        view_269: "i64[2401]" = torch.ops.aten.view.default(arg228_1, [-1]);  arg228_1 = None
        index_19: "f32[2401, 16]" = torch.ops.aten.index.Tensor(arg229_1, [view_269]);  arg229_1 = None
        view_270: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_19, _shape_param_265);  index_19 = _shape_param_265 = None
        permute_84: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_270, [2, 0, 1]);  view_270 = None
        clone_34: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
        unsqueeze_15: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_34, 0);  clone_34 = None
        add_42: "f32[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_268, unsqueeze_15);  view_268 = unsqueeze_15 = None
        sub_58: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(add_42, arg230_1);  add_42 = arg230_1 = None
        exp_15: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_58);  sub_58 = None
        div_41: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_15, arg231_1);  exp_15 = arg231_1 = None
        mul_205: "f32[512, 16, 49, 49]" = torch.ops.aten.mul.Tensor(convert_element_type_199, div_41);  convert_element_type_199 = None
        sum_114: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_205, [-1], True)
        neg_7: "f32[512, 16, 49, 49]" = torch.ops.aten.neg.default(div_41);  div_41 = None
        fma_7: "f32[512, 16, 49, 49]" = torch.ops.prims.fma.default(neg_7, sum_114, mul_205);  neg_7 = sum_114 = mul_205 = None
        convert_element_type_200: "bf16[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(fma_7, torch.bfloat16)
        sum_115: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_7, [0], True, dtype = torch.float32);  fma_7 = None
        squeeze_7: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_115, 0);  sum_115 = None
        permute_85: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_7, [1, 2, 0]);  squeeze_7 = None
        view_271: "f32[2401, 16]" = torch.ops.aten.view.default(permute_85, _shape_param_266);  permute_85 = _shape_param_266 = None
        index_put_7: "f32[169, 16]" = torch.ops.aten.index_put.default(full_1, [view_269], view_271, True);  view_269 = view_271 = None
        view_272: "bf16[8192, 49, 49]" = torch.ops.aten.view.default(convert_element_type_200, _shape_param_267);  convert_element_type_200 = _shape_param_267 = None
        bmm_30: "bf16[8192, 32, 49]" = torch.ops.aten.bmm.default(arg232_1, view_272);  arg232_1 = None
        bmm_31: "bf16[8192, 49, 32]" = torch.ops.aten.bmm.default(view_272, arg233_1);  view_272 = arg233_1 = None
        view_273: "bf16[512, 16, 32, 49]" = torch.ops.aten.view.default(bmm_30, _shape_param_268);  bmm_30 = _shape_param_268 = None
        view_274: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(bmm_31, _shape_param_269);  bmm_31 = _shape_param_269 = None
        permute_86: "bf16[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_273, [0, 1, 3, 2]);  view_273 = None
        mul_206: "bf16[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_274, 0.1767766952966369);  view_274 = None
        cat_7: "bf16[1536, 16, 49, 32]" = torch.ops.aten.cat.default([mul_206, permute_86, view_266]);  mul_206 = permute_86 = view_266 = None
        view_275: "bf16[3, 512, 16, 49, 32]" = torch.ops.aten.view.default(cat_7, _shape_param_270);  cat_7 = _shape_param_270 = None
        permute_87: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.permute.default(view_275, [1, 3, 0, 2, 4]);  view_275 = None
        clone_35: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_87, memory_format = torch.contiguous_format);  permute_87 = None
        view_276: "bf16[512, 49, 1536]" = torch.ops.aten.view.default(clone_35, _shape_param_271);  clone_35 = _shape_param_271 = None
        view_277: "bf16[25088, 1536]" = torch.ops.aten.view.default(view_276, _shape_param_272);  view_276 = _shape_param_272 = None
        mm_65: "bf16[25088, 512]" = torch.ops.aten.mm.default(view_277, arg234_1);  arg234_1 = None
        permute_88: "bf16[1536, 25088]" = torch.ops.aten.permute.default(view_277, [1, 0])
        mm_66: "bf16[1536, 512]" = torch.ops.aten.mm.default(permute_88, arg235_1);  permute_88 = arg235_1 = None
        sum_116: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_277, [0], True, dtype = torch.float32);  view_277 = None
        view_278: "f32[1536]" = torch.ops.aten.view.default(sum_116, _shape_param_273);  sum_116 = _shape_param_273 = None
        convert_element_type_201: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_278, torch.bfloat16);  view_278 = None
        view_279: "bf16[512, 49, 512]" = torch.ops.aten.view.default(mm_65, _shape_param_274);  mm_65 = _shape_param_274 = None
        convert_element_type_202: "f32[512, 49, 512]" = torch.ops.prims.convert_element_type.default(view_279, torch.float32);  view_279 = None
        convert_element_type_203: "f32[1536, 512]" = torch.ops.prims.convert_element_type.default(mm_66, torch.float32);  mm_66 = None
        convert_element_type_204: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_201, torch.float32);  convert_element_type_201 = None
        view_280: "f32[512, 7, 7, 512]" = torch.ops.aten.view.default(convert_element_type_202, _shape_param_275);  convert_element_type_202 = _shape_param_275 = None
        view_281: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_280, _shape_param_276);  view_280 = _shape_param_276 = None
        permute_89: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_281, [0, 1, 3, 2, 4, 5]);  view_281 = None
        clone_36: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_89, memory_format = torch.contiguous_format);  permute_89 = None
        view_282: "f32[128, 14, 14, 512]" = torch.ops.aten.view.default(clone_36, _shape_param_277);  clone_36 = _shape_param_277 = None
        mul_207: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_282, arg236_1);  arg236_1 = None
        mul_208: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_207, 512)
        sum_117: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_207, [3], True)
        convert_element_type_205: "f32[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(arg237_1, torch.float32);  arg237_1 = None
        sub_59: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_205, arg238_1);  convert_element_type_205 = arg238_1 = None
        mul_209: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_59, arg239_1);  sub_59 = None
        mul_210: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_207, mul_209);  mul_207 = None
        sum_118: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_210, [3], True);  mul_210 = None
        mul_211: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_209, sum_118);  sum_118 = None
        sub_60: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_208, sum_117);  mul_208 = sum_117 = None
        sub_61: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_60, mul_211);  sub_60 = mul_211 = None
        div_42: "f32[128, 14, 14, 1]" = torch.ops.aten.div.Tensor(arg239_1, 512);  arg239_1 = None
        mul_212: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_42, sub_61);  div_42 = sub_61 = None
        mul_213: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_282, mul_209);  mul_209 = None
        sum_119: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_213, [0, 1, 2]);  mul_213 = None
        sum_120: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_282, [0, 1, 2]);  view_282 = None
        convert_element_type_206: "bf16[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(mul_212, torch.bfloat16);  mul_212 = None
        add_43: "bf16[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_257, convert_element_type_206);  view_257 = convert_element_type_206 = None
        view_283: "bf16[128, 196, 512]" = torch.ops.aten.view.default(add_43, _shape_param_278);  add_43 = _shape_param_278 = None
        convert_element_type_207: "bf16[128, 1, 1]" = torch.ops.prims.convert_element_type.default(arg240_1, torch.bfloat16);  arg240_1 = None
        div_43: "bf16[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_207, 0.9347826093435287);  convert_element_type_207 = None
        mul_214: "bf16[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_283, div_43);  div_43 = None
        view_284: "bf16[25088, 512]" = torch.ops.aten.view.default(mul_214, _shape_param_279);  mul_214 = _shape_param_279 = None
        mm_67: "bf16[25088, 2048]" = torch.ops.aten.mm.default(view_284, arg241_1);  arg241_1 = None
        permute_90: "bf16[512, 25088]" = torch.ops.aten.permute.default(view_284, [1, 0])
        mm_68: "bf16[512, 2048]" = torch.ops.aten.mm.default(permute_90, arg242_1);  permute_90 = arg242_1 = None
        sum_121: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_284, [0], True, dtype = torch.float32);  view_284 = None
        view_285: "f32[512]" = torch.ops.aten.view.default(sum_121, _shape_param_280);  sum_121 = _shape_param_280 = None
        convert_element_type_208: "bf16[512]" = torch.ops.prims.convert_element_type.default(view_285, torch.bfloat16);  view_285 = None
        view_286: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(mm_67, _shape_param_281);  mm_67 = _shape_param_281 = None
        convert_element_type_209: "f32[512, 2048]" = torch.ops.prims.convert_element_type.default(mm_68, torch.float32);  mm_68 = None
        convert_element_type_210: "f32[512]" = torch.ops.prims.convert_element_type.default(convert_element_type_208, torch.float32);  convert_element_type_208 = None
        convert_element_type_211: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_286, torch.float32);  view_286 = None
        view_287: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(arg243_1, _shape_param_282);  arg243_1 = _shape_param_282 = None
        convert_element_type_212: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_287, torch.float32);  view_287 = None
        mul_215: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_212, 0.7071067811865476)
        erf_8: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_215);  mul_215 = None
        add_44: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_216: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(add_44, 0.5);  add_44 = None
        mul_217: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_212, convert_element_type_212)
        mul_218: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_217, -0.5);  mul_217 = None
        exp_16: "f32[128, 196, 2048]" = torch.ops.aten.exp.default(mul_218);  mul_218 = None
        mul_219: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_16, 0.3989422804014327);  exp_16 = None
        mul_220: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_212, mul_219);  convert_element_type_212 = mul_219 = None
        add_45: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(mul_216, mul_220);  mul_216 = mul_220 = None
        mul_221: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_211, add_45);  convert_element_type_211 = add_45 = None
        convert_element_type_213: "bf16[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(mul_221, torch.bfloat16);  mul_221 = None
        view_288: "bf16[25088, 2048]" = torch.ops.aten.view.default(convert_element_type_213, _shape_param_283);  convert_element_type_213 = _shape_param_283 = None
        mm_69: "bf16[25088, 512]" = torch.ops.aten.mm.default(view_288, arg244_1);  arg244_1 = None
        permute_91: "bf16[2048, 25088]" = torch.ops.aten.permute.default(view_288, [1, 0])
        mm_70: "bf16[2048, 512]" = torch.ops.aten.mm.default(permute_91, arg245_1);  permute_91 = arg245_1 = None
        sum_122: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_288, [0], True, dtype = torch.float32);  view_288 = None
        view_289: "f32[2048]" = torch.ops.aten.view.default(sum_122, _shape_param_284);  sum_122 = _shape_param_284 = None
        convert_element_type_214: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_289, torch.bfloat16);  view_289 = None
        view_290: "bf16[128, 196, 512]" = torch.ops.aten.view.default(mm_69, _shape_param_285);  mm_69 = _shape_param_285 = None
        convert_element_type_215: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(view_290, torch.float32);  view_290 = None
        convert_element_type_216: "f32[2048, 512]" = torch.ops.prims.convert_element_type.default(mm_70, torch.float32);  mm_70 = None
        convert_element_type_217: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_214, torch.float32);  convert_element_type_214 = None
        mul_222: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_215, arg246_1);  arg246_1 = None
        mul_223: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_222, 512)
        sum_123: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_222, [2], True)
        convert_element_type_218: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(arg247_1, torch.float32);  arg247_1 = None
        sub_62: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_218, arg248_1);  convert_element_type_218 = arg248_1 = None
        mul_224: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_62, arg249_1);  sub_62 = None
        mul_225: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_222, mul_224);  mul_222 = None
        sum_124: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_225, [2], True);  mul_225 = None
        mul_226: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_224, sum_124);  sum_124 = None
        sub_63: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(mul_223, sum_123);  mul_223 = sum_123 = None
        sub_64: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(sub_63, mul_226);  sub_63 = mul_226 = None
        div_44: "f32[128, 196, 1]" = torch.ops.aten.div.Tensor(arg249_1, 512);  arg249_1 = None
        mul_227: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(div_44, sub_64);  div_44 = sub_64 = None
        mul_228: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_215, mul_224);  mul_224 = None
        sum_125: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_228, [0, 1]);  mul_228 = None
        sum_126: "f32[512]" = torch.ops.aten.sum.dim_IntList(convert_element_type_215, [0, 1]);  convert_element_type_215 = None
        convert_element_type_219: "bf16[128, 196, 512]" = torch.ops.prims.convert_element_type.default(mul_227, torch.bfloat16);  mul_227 = None
        add_46: "bf16[128, 196, 512]" = torch.ops.aten.add.Tensor(view_283, convert_element_type_219);  view_283 = convert_element_type_219 = None
        view_291: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(add_46, _shape_param_286);  add_46 = _shape_param_286 = None
        convert_element_type_220: "bf16[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(arg250_1, torch.bfloat16);  arg250_1 = None
        div_45: "bf16[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_220, 0.9347826093435287);  convert_element_type_220 = None
        mul_229: "bf16[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_291, div_45);  div_45 = None
        index_20: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(mul_229, [None, None, fmod]);  mul_229 = None
        index_21: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_20, [None, fmod]);  index_20 = None
        view_292: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(index_21, _shape_param_287);  index_21 = _shape_param_287 = None
        permute_92: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_292, [0, 1, 3, 2, 4, 5]);  view_292 = None
        clone_37: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_92, memory_format = torch.contiguous_format);  permute_92 = None
        view_293: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(clone_37, _shape_param_288);  clone_37 = _shape_param_288 = None
        view_294: "bf16[512, 49, 512]" = torch.ops.aten.view.default(view_293, _shape_param_289);  view_293 = _shape_param_289 = None
        view_295: "bf16[25088, 512]" = torch.ops.aten.view.default(view_294, _shape_param_290);  view_294 = _shape_param_290 = None
        mm_71: "bf16[25088, 512]" = torch.ops.aten.mm.default(view_295, arg251_1);  arg251_1 = None
        permute_93: "bf16[512, 25088]" = torch.ops.aten.permute.default(view_295, [1, 0])
        mm_72: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_93, arg252_1);  permute_93 = arg252_1 = None
        sum_127: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_295, [0], True, dtype = torch.float32);  view_295 = None
        view_296: "f32[512]" = torch.ops.aten.view.default(sum_127, _shape_param_291);  sum_127 = _shape_param_291 = None
        convert_element_type_221: "bf16[512]" = torch.ops.prims.convert_element_type.default(view_296, torch.bfloat16);  view_296 = None
        view_297: "bf16[512, 49, 512]" = torch.ops.aten.view.default(mm_71, _shape_param_292);  mm_71 = _shape_param_292 = None
        convert_element_type_222: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_72, torch.float32);  mm_72 = None
        convert_element_type_223: "f32[512]" = torch.ops.prims.convert_element_type.default(convert_element_type_221, torch.float32);  convert_element_type_221 = None
        view_298: "bf16[512, 49, 16, 32]" = torch.ops.aten.view.default(view_297, _shape_param_293);  view_297 = _shape_param_293 = None
        permute_94: "bf16[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_298, [0, 2, 1, 3]);  view_298 = None
        clone_38: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(permute_94, memory_format = torch.contiguous_format);  permute_94 = None
        view_299: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_38, _shape_param_294);  clone_38 = _shape_param_294 = None
        bmm_32: "bf16[8192, 49, 32]" = torch.ops.aten.bmm.default(arg253_1, view_299);  arg253_1 = None
        bmm_33: "bf16[8192, 49, 49]" = torch.ops.aten.bmm.default(view_299, arg254_1);  view_299 = arg254_1 = None
        view_300: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(bmm_32, _shape_param_295);  bmm_32 = _shape_param_295 = None
        view_301: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(bmm_33, _shape_param_296);  bmm_33 = _shape_param_296 = None
        convert_element_type_224: "f32[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(view_301, torch.float32);  view_301 = None
        slice_13: "bf16[8192, 49, 56]" = torch.ops.aten.slice.Tensor(arg255_1, 1, 0, -7);  arg255_1 = None
        slice_14: "bf16[8192, 49, 49]" = torch.ops.aten.slice.Tensor(slice_13, 2, 0, -7);  slice_13 = None
        view_302: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(slice_14, _shape_param_297);  slice_14 = _shape_param_297 = None
        view_303: "i64[2401]" = torch.ops.aten.view.default(arg256_1, [-1]);  arg256_1 = None
        index_22: "f32[2401, 16]" = torch.ops.aten.index.Tensor(arg257_1, [view_303]);  arg257_1 = None
        view_304: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_22, _shape_param_298);  index_22 = _shape_param_298 = None
        permute_95: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_304, [2, 0, 1]);  view_304 = None
        clone_39: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
        unsqueeze_16: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_39, 0);  clone_39 = None
        add_47: "f32[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_302, unsqueeze_16);  view_302 = unsqueeze_16 = None
        view_305: "f32[128, 4, 16, 49, 49]" = torch.ops.aten.view.default(add_47, _shape_param_299);  add_47 = _shape_param_299 = None
        unsqueeze_17: "f32[4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(arg258_1, 1);  arg258_1 = None
        unsqueeze_18: "f32[1, 4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_17, 0);  unsqueeze_17 = None
        add_48: "f32[128, 4, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_305, unsqueeze_18);  view_305 = unsqueeze_18 = None
        view_306: "f32[512, 16, 49, 49]" = torch.ops.aten.view.default(add_48, _shape_param_300);  add_48 = _shape_param_300 = None
        sub_65: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(view_306, arg259_1);  view_306 = arg259_1 = None
        exp_17: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_65);  sub_65 = None
        div_46: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_17, arg260_1);  exp_17 = arg260_1 = None
        mul_230: "f32[512, 16, 49, 49]" = torch.ops.aten.mul.Tensor(convert_element_type_224, div_46);  convert_element_type_224 = None
        sum_128: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_230, [-1], True)
        neg_8: "f32[512, 16, 49, 49]" = torch.ops.aten.neg.default(div_46);  div_46 = None
        fma_8: "f32[512, 16, 49, 49]" = torch.ops.prims.fma.default(neg_8, sum_128, mul_230);  neg_8 = sum_128 = mul_230 = None
        convert_element_type_225: "bf16[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(fma_8, torch.bfloat16)
        sum_129: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_8, [0], True, dtype = torch.float32);  fma_8 = None
        squeeze_8: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_129, 0);  sum_129 = None
        permute_96: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_8, [1, 2, 0]);  squeeze_8 = None
        view_307: "f32[2401, 16]" = torch.ops.aten.view.default(permute_96, _shape_param_301);  permute_96 = _shape_param_301 = None
        index_put_8: "f32[169, 16]" = torch.ops.aten.index_put.default(full_1, [view_303], view_307, True);  view_303 = view_307 = None
        view_308: "bf16[8192, 49, 49]" = torch.ops.aten.view.default(convert_element_type_225, _shape_param_302);  convert_element_type_225 = _shape_param_302 = None
        bmm_34: "bf16[8192, 32, 49]" = torch.ops.aten.bmm.default(arg261_1, view_308);  arg261_1 = None
        bmm_35: "bf16[8192, 49, 32]" = torch.ops.aten.bmm.default(view_308, arg262_1);  view_308 = arg262_1 = None
        view_309: "bf16[512, 16, 32, 49]" = torch.ops.aten.view.default(bmm_34, _shape_param_303);  bmm_34 = _shape_param_303 = None
        view_310: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(bmm_35, _shape_param_304);  bmm_35 = _shape_param_304 = None
        permute_97: "bf16[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_309, [0, 1, 3, 2]);  view_309 = None
        mul_231: "bf16[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_310, 0.1767766952966369);  view_310 = None
        cat_8: "bf16[1536, 16, 49, 32]" = torch.ops.aten.cat.default([mul_231, permute_97, view_300]);  mul_231 = permute_97 = view_300 = None
        view_311: "bf16[3, 512, 16, 49, 32]" = torch.ops.aten.view.default(cat_8, _shape_param_305);  cat_8 = _shape_param_305 = None
        permute_98: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.permute.default(view_311, [1, 3, 0, 2, 4]);  view_311 = None
        clone_40: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_98, memory_format = torch.contiguous_format);  permute_98 = None
        view_312: "bf16[512, 49, 1536]" = torch.ops.aten.view.default(clone_40, _shape_param_306);  clone_40 = _shape_param_306 = None
        view_313: "bf16[25088, 1536]" = torch.ops.aten.view.default(view_312, _shape_param_307);  view_312 = _shape_param_307 = None
        mm_73: "bf16[25088, 512]" = torch.ops.aten.mm.default(view_313, arg263_1);  arg263_1 = None
        permute_99: "bf16[1536, 25088]" = torch.ops.aten.permute.default(view_313, [1, 0])
        mm_74: "bf16[1536, 512]" = torch.ops.aten.mm.default(permute_99, arg264_1);  permute_99 = arg264_1 = None
        sum_130: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_313, [0], True, dtype = torch.float32);  view_313 = None
        view_314: "f32[1536]" = torch.ops.aten.view.default(sum_130, _shape_param_308);  sum_130 = _shape_param_308 = None
        convert_element_type_226: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_314, torch.bfloat16);  view_314 = None
        view_315: "bf16[512, 49, 512]" = torch.ops.aten.view.default(mm_73, _shape_param_309);  mm_73 = _shape_param_309 = None
        convert_element_type_227: "f32[512, 49, 512]" = torch.ops.prims.convert_element_type.default(view_315, torch.float32);  view_315 = None
        convert_element_type_228: "f32[1536, 512]" = torch.ops.prims.convert_element_type.default(mm_74, torch.float32);  mm_74 = None
        convert_element_type_229: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_226, torch.float32);  convert_element_type_226 = None
        view_316: "f32[512, 7, 7, 512]" = torch.ops.aten.view.default(convert_element_type_227, _shape_param_310);  convert_element_type_227 = _shape_param_310 = None
        view_317: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_316, _shape_param_311);  view_316 = _shape_param_311 = None
        permute_100: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_317, [0, 1, 3, 2, 4, 5]);  view_317 = None
        clone_41: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_100, memory_format = torch.contiguous_format);  permute_100 = None
        view_318: "f32[128, 14, 14, 512]" = torch.ops.aten.view.default(clone_41, _shape_param_312);  clone_41 = _shape_param_312 = None
        index_23: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(view_318, [None, None, arg93_1]);  view_318 = None
        index_24: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_23, [None, arg93_1]);  index_23 = None
        mul_232: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(index_24, arg265_1);  arg265_1 = None
        mul_233: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_232, 512)
        sum_131: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_232, [3], True)
        convert_element_type_230: "f32[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(arg266_1, torch.float32);  arg266_1 = None
        sub_66: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_230, arg267_1);  convert_element_type_230 = arg267_1 = None
        mul_234: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_66, arg268_1);  sub_66 = None
        mul_235: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_232, mul_234);  mul_232 = None
        sum_132: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_235, [3], True);  mul_235 = None
        mul_236: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_234, sum_132);  sum_132 = None
        sub_67: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_233, sum_131);  mul_233 = sum_131 = None
        sub_68: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_67, mul_236);  sub_67 = mul_236 = None
        div_47: "f32[128, 14, 14, 1]" = torch.ops.aten.div.Tensor(arg268_1, 512);  arg268_1 = None
        mul_237: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_47, sub_68);  div_47 = sub_68 = None
        mul_238: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(index_24, mul_234);  mul_234 = None
        sum_133: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_238, [0, 1, 2]);  mul_238 = None
        sum_134: "f32[512]" = torch.ops.aten.sum.dim_IntList(index_24, [0, 1, 2]);  index_24 = None
        convert_element_type_231: "bf16[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(mul_237, torch.bfloat16);  mul_237 = None
        add_49: "bf16[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_291, convert_element_type_231);  view_291 = convert_element_type_231 = None
        view_319: "bf16[128, 196, 512]" = torch.ops.aten.view.default(add_49, _shape_param_313);  add_49 = _shape_param_313 = None
        convert_element_type_232: "bf16[128, 1, 1]" = torch.ops.prims.convert_element_type.default(arg269_1, torch.bfloat16);  arg269_1 = None
        div_48: "bf16[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_232, 0.9391304366290569);  convert_element_type_232 = None
        mul_239: "bf16[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_319, div_48);  div_48 = None
        view_320: "bf16[25088, 512]" = torch.ops.aten.view.default(mul_239, _shape_param_314);  mul_239 = _shape_param_314 = None
        mm_75: "bf16[25088, 2048]" = torch.ops.aten.mm.default(view_320, arg270_1);  arg270_1 = None
        permute_101: "bf16[512, 25088]" = torch.ops.aten.permute.default(view_320, [1, 0])
        mm_76: "bf16[512, 2048]" = torch.ops.aten.mm.default(permute_101, arg271_1);  permute_101 = arg271_1 = None
        sum_135: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_320, [0], True, dtype = torch.float32);  view_320 = None
        view_321: "f32[512]" = torch.ops.aten.view.default(sum_135, _shape_param_315);  sum_135 = _shape_param_315 = None
        convert_element_type_233: "bf16[512]" = torch.ops.prims.convert_element_type.default(view_321, torch.bfloat16);  view_321 = None
        view_322: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(mm_75, _shape_param_316);  mm_75 = _shape_param_316 = None
        convert_element_type_234: "f32[512, 2048]" = torch.ops.prims.convert_element_type.default(mm_76, torch.float32);  mm_76 = None
        convert_element_type_235: "f32[512]" = torch.ops.prims.convert_element_type.default(convert_element_type_233, torch.float32);  convert_element_type_233 = None
        convert_element_type_236: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_322, torch.float32);  view_322 = None
        view_323: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(arg272_1, _shape_param_317);  arg272_1 = _shape_param_317 = None
        convert_element_type_237: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_323, torch.float32);  view_323 = None
        mul_240: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_237, 0.7071067811865476)
        erf_9: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_240);  mul_240 = None
        add_50: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_241: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(add_50, 0.5);  add_50 = None
        mul_242: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_237, convert_element_type_237)
        mul_243: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_242, -0.5);  mul_242 = None
        exp_18: "f32[128, 196, 2048]" = torch.ops.aten.exp.default(mul_243);  mul_243 = None
        mul_244: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_18, 0.3989422804014327);  exp_18 = None
        mul_245: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_237, mul_244);  convert_element_type_237 = mul_244 = None
        add_51: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(mul_241, mul_245);  mul_241 = mul_245 = None
        mul_246: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_236, add_51);  convert_element_type_236 = add_51 = None
        convert_element_type_238: "bf16[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(mul_246, torch.bfloat16);  mul_246 = None
        view_324: "bf16[25088, 2048]" = torch.ops.aten.view.default(convert_element_type_238, _shape_param_318);  convert_element_type_238 = _shape_param_318 = None
        mm_77: "bf16[25088, 512]" = torch.ops.aten.mm.default(view_324, arg273_1);  arg273_1 = None
        permute_102: "bf16[2048, 25088]" = torch.ops.aten.permute.default(view_324, [1, 0])
        mm_78: "bf16[2048, 512]" = torch.ops.aten.mm.default(permute_102, arg274_1);  permute_102 = arg274_1 = None
        sum_136: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_324, [0], True, dtype = torch.float32);  view_324 = None
        view_325: "f32[2048]" = torch.ops.aten.view.default(sum_136, _shape_param_319);  sum_136 = _shape_param_319 = None
        convert_element_type_239: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_325, torch.bfloat16);  view_325 = None
        view_326: "bf16[128, 196, 512]" = torch.ops.aten.view.default(mm_77, _shape_param_320);  mm_77 = _shape_param_320 = None
        convert_element_type_240: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(view_326, torch.float32);  view_326 = None
        convert_element_type_241: "f32[2048, 512]" = torch.ops.prims.convert_element_type.default(mm_78, torch.float32);  mm_78 = None
        convert_element_type_242: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_239, torch.float32);  convert_element_type_239 = None
        mul_247: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_240, arg275_1);  arg275_1 = None
        mul_248: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_247, 512)
        sum_137: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_247, [2], True)
        convert_element_type_243: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(arg276_1, torch.float32);  arg276_1 = None
        sub_69: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_243, arg277_1);  convert_element_type_243 = arg277_1 = None
        mul_249: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_69, arg278_1);  sub_69 = None
        mul_250: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_247, mul_249);  mul_247 = None
        sum_138: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_250, [2], True);  mul_250 = None
        mul_251: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_249, sum_138);  sum_138 = None
        sub_70: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(mul_248, sum_137);  mul_248 = sum_137 = None
        sub_71: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(sub_70, mul_251);  sub_70 = mul_251 = None
        div_49: "f32[128, 196, 1]" = torch.ops.aten.div.Tensor(arg278_1, 512);  arg278_1 = None
        mul_252: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(div_49, sub_71);  div_49 = sub_71 = None
        mul_253: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_240, mul_249);  mul_249 = None
        sum_139: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_253, [0, 1]);  mul_253 = None
        sum_140: "f32[512]" = torch.ops.aten.sum.dim_IntList(convert_element_type_240, [0, 1]);  convert_element_type_240 = None
        convert_element_type_244: "bf16[128, 196, 512]" = torch.ops.prims.convert_element_type.default(mul_252, torch.bfloat16);  mul_252 = None
        add_52: "bf16[128, 196, 512]" = torch.ops.aten.add.Tensor(view_319, convert_element_type_244);  view_319 = convert_element_type_244 = None
        view_327: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(add_52, _shape_param_321);  add_52 = _shape_param_321 = None
        convert_element_type_245: "bf16[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(arg279_1, torch.bfloat16);  arg279_1 = None
        div_50: "bf16[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_245, 0.9391304366290569);  convert_element_type_245 = None
        mul_254: "bf16[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_327, div_50);  div_50 = None
        view_328: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(mul_254, _shape_param_322);  mul_254 = _shape_param_322 = None
        permute_103: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_328, [0, 1, 3, 2, 4, 5]);  view_328 = None
        clone_42: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_103, memory_format = torch.contiguous_format);  permute_103 = None
        view_329: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(clone_42, _shape_param_323);  clone_42 = _shape_param_323 = None
        view_330: "bf16[512, 49, 512]" = torch.ops.aten.view.default(view_329, _shape_param_324);  view_329 = _shape_param_324 = None
        view_331: "bf16[25088, 512]" = torch.ops.aten.view.default(view_330, _shape_param_325);  view_330 = _shape_param_325 = None
        mm_79: "bf16[25088, 512]" = torch.ops.aten.mm.default(view_331, arg280_1);  arg280_1 = None
        permute_104: "bf16[512, 25088]" = torch.ops.aten.permute.default(view_331, [1, 0])
        mm_80: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_104, arg281_1);  permute_104 = arg281_1 = None
        sum_141: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_331, [0], True, dtype = torch.float32);  view_331 = None
        view_332: "f32[512]" = torch.ops.aten.view.default(sum_141, _shape_param_326);  sum_141 = _shape_param_326 = None
        convert_element_type_246: "bf16[512]" = torch.ops.prims.convert_element_type.default(view_332, torch.bfloat16);  view_332 = None
        view_333: "bf16[512, 49, 512]" = torch.ops.aten.view.default(mm_79, _shape_param_327);  mm_79 = _shape_param_327 = None
        convert_element_type_247: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_80, torch.float32);  mm_80 = None
        convert_element_type_248: "f32[512]" = torch.ops.prims.convert_element_type.default(convert_element_type_246, torch.float32);  convert_element_type_246 = None
        view_334: "bf16[512, 49, 16, 32]" = torch.ops.aten.view.default(view_333, _shape_param_328);  view_333 = _shape_param_328 = None
        permute_105: "bf16[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_334, [0, 2, 1, 3]);  view_334 = None
        clone_43: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(permute_105, memory_format = torch.contiguous_format);  permute_105 = None
        view_335: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_43, _shape_param_329);  clone_43 = _shape_param_329 = None
        bmm_36: "bf16[8192, 49, 32]" = torch.ops.aten.bmm.default(arg282_1, view_335);  arg282_1 = None
        bmm_37: "bf16[8192, 49, 49]" = torch.ops.aten.bmm.default(view_335, arg283_1);  view_335 = arg283_1 = None
        view_336: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(bmm_36, _shape_param_330);  bmm_36 = _shape_param_330 = None
        view_337: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(bmm_37, _shape_param_331);  bmm_37 = _shape_param_331 = None
        convert_element_type_249: "f32[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(view_337, torch.float32);  view_337 = None
        slice_15: "bf16[8192, 49, 56]" = torch.ops.aten.slice.Tensor(arg284_1, 1, 0, -7);  arg284_1 = None
        slice_16: "bf16[8192, 49, 49]" = torch.ops.aten.slice.Tensor(slice_15, 2, 0, -7);  slice_15 = None
        view_338: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(slice_16, _shape_param_332);  slice_16 = _shape_param_332 = None
        view_339: "i64[2401]" = torch.ops.aten.view.default(arg285_1, [-1]);  arg285_1 = None
        index_25: "f32[2401, 16]" = torch.ops.aten.index.Tensor(arg286_1, [view_339]);  arg286_1 = None
        view_340: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_25, _shape_param_333);  index_25 = _shape_param_333 = None
        permute_106: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_340, [2, 0, 1]);  view_340 = None
        clone_44: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
        unsqueeze_19: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_44, 0);  clone_44 = None
        add_53: "f32[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_338, unsqueeze_19);  view_338 = unsqueeze_19 = None
        sub_72: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(add_53, arg287_1);  add_53 = arg287_1 = None
        exp_19: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_72);  sub_72 = None
        div_51: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_19, arg288_1);  exp_19 = arg288_1 = None
        mul_255: "f32[512, 16, 49, 49]" = torch.ops.aten.mul.Tensor(convert_element_type_249, div_51);  convert_element_type_249 = None
        sum_142: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_255, [-1], True)
        neg_9: "f32[512, 16, 49, 49]" = torch.ops.aten.neg.default(div_51);  div_51 = None
        fma_9: "f32[512, 16, 49, 49]" = torch.ops.prims.fma.default(neg_9, sum_142, mul_255);  neg_9 = sum_142 = mul_255 = None
        convert_element_type_250: "bf16[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(fma_9, torch.bfloat16)
        sum_143: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_9, [0], True, dtype = torch.float32);  fma_9 = None
        squeeze_9: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_143, 0);  sum_143 = None
        permute_107: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_9, [1, 2, 0]);  squeeze_9 = None
        view_341: "f32[2401, 16]" = torch.ops.aten.view.default(permute_107, _shape_param_334);  permute_107 = _shape_param_334 = None
        index_put_9: "f32[169, 16]" = torch.ops.aten.index_put.default(full_1, [view_339], view_341, True);  view_339 = view_341 = None
        view_342: "bf16[8192, 49, 49]" = torch.ops.aten.view.default(convert_element_type_250, _shape_param_335);  convert_element_type_250 = _shape_param_335 = None
        bmm_38: "bf16[8192, 32, 49]" = torch.ops.aten.bmm.default(arg289_1, view_342);  arg289_1 = None
        bmm_39: "bf16[8192, 49, 32]" = torch.ops.aten.bmm.default(view_342, arg290_1);  view_342 = arg290_1 = None
        view_343: "bf16[512, 16, 32, 49]" = torch.ops.aten.view.default(bmm_38, _shape_param_336);  bmm_38 = _shape_param_336 = None
        view_344: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(bmm_39, _shape_param_337);  bmm_39 = _shape_param_337 = None
        permute_108: "bf16[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_343, [0, 1, 3, 2]);  view_343 = None
        mul_256: "bf16[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_344, 0.1767766952966369);  view_344 = None
        cat_9: "bf16[1536, 16, 49, 32]" = torch.ops.aten.cat.default([mul_256, permute_108, view_336]);  mul_256 = permute_108 = view_336 = None
        view_345: "bf16[3, 512, 16, 49, 32]" = torch.ops.aten.view.default(cat_9, _shape_param_338);  cat_9 = _shape_param_338 = None
        permute_109: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.permute.default(view_345, [1, 3, 0, 2, 4]);  view_345 = None
        clone_45: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_109, memory_format = torch.contiguous_format);  permute_109 = None
        view_346: "bf16[512, 49, 1536]" = torch.ops.aten.view.default(clone_45, _shape_param_339);  clone_45 = _shape_param_339 = None
        view_347: "bf16[25088, 1536]" = torch.ops.aten.view.default(view_346, _shape_param_340);  view_346 = _shape_param_340 = None
        mm_81: "bf16[25088, 512]" = torch.ops.aten.mm.default(view_347, arg291_1);  arg291_1 = None
        permute_110: "bf16[1536, 25088]" = torch.ops.aten.permute.default(view_347, [1, 0])
        mm_82: "bf16[1536, 512]" = torch.ops.aten.mm.default(permute_110, arg292_1);  permute_110 = arg292_1 = None
        sum_144: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_347, [0], True, dtype = torch.float32);  view_347 = None
        view_348: "f32[1536]" = torch.ops.aten.view.default(sum_144, _shape_param_341);  sum_144 = _shape_param_341 = None
        convert_element_type_251: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_348, torch.bfloat16);  view_348 = None
        view_349: "bf16[512, 49, 512]" = torch.ops.aten.view.default(mm_81, _shape_param_342);  mm_81 = _shape_param_342 = None
        convert_element_type_252: "f32[512, 49, 512]" = torch.ops.prims.convert_element_type.default(view_349, torch.float32);  view_349 = None
        convert_element_type_253: "f32[1536, 512]" = torch.ops.prims.convert_element_type.default(mm_82, torch.float32);  mm_82 = None
        convert_element_type_254: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_251, torch.float32);  convert_element_type_251 = None
        view_350: "f32[512, 7, 7, 512]" = torch.ops.aten.view.default(convert_element_type_252, _shape_param_343);  convert_element_type_252 = _shape_param_343 = None
        view_351: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_350, _shape_param_344);  view_350 = _shape_param_344 = None
        permute_111: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_351, [0, 1, 3, 2, 4, 5]);  view_351 = None
        clone_46: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_111, memory_format = torch.contiguous_format);  permute_111 = None
        view_352: "f32[128, 14, 14, 512]" = torch.ops.aten.view.default(clone_46, _shape_param_345);  clone_46 = _shape_param_345 = None
        mul_257: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_352, arg293_1);  arg293_1 = None
        mul_258: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_257, 512)
        sum_145: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_257, [3], True)
        convert_element_type_255: "f32[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(arg294_1, torch.float32);  arg294_1 = None
        sub_73: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_255, arg295_1);  convert_element_type_255 = arg295_1 = None
        mul_259: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_73, arg296_1);  sub_73 = None
        mul_260: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_257, mul_259);  mul_257 = None
        sum_146: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_260, [3], True);  mul_260 = None
        mul_261: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_259, sum_146);  sum_146 = None
        sub_74: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_258, sum_145);  mul_258 = sum_145 = None
        sub_75: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_74, mul_261);  sub_74 = mul_261 = None
        div_52: "f32[128, 14, 14, 1]" = torch.ops.aten.div.Tensor(arg296_1, 512);  arg296_1 = None
        mul_262: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_52, sub_75);  div_52 = sub_75 = None
        mul_263: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_352, mul_259);  mul_259 = None
        sum_147: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_263, [0, 1, 2]);  mul_263 = None
        sum_148: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_352, [0, 1, 2]);  view_352 = None
        convert_element_type_256: "bf16[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(mul_262, torch.bfloat16);  mul_262 = None
        add_54: "bf16[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_327, convert_element_type_256);  view_327 = convert_element_type_256 = None
        view_353: "bf16[128, 196, 512]" = torch.ops.aten.view.default(add_54, _shape_param_346);  add_54 = _shape_param_346 = None
        convert_element_type_257: "bf16[128, 1, 1]" = torch.ops.prims.convert_element_type.default(arg297_1, torch.bfloat16);  arg297_1 = None
        div_53: "bf16[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_257, 0.9434782639145851);  convert_element_type_257 = None
        mul_264: "bf16[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_353, div_53);  div_53 = None
        view_354: "bf16[25088, 512]" = torch.ops.aten.view.default(mul_264, _shape_param_347);  mul_264 = _shape_param_347 = None
        mm_83: "bf16[25088, 2048]" = torch.ops.aten.mm.default(view_354, arg298_1);  arg298_1 = None
        permute_112: "bf16[512, 25088]" = torch.ops.aten.permute.default(view_354, [1, 0])
        mm_84: "bf16[512, 2048]" = torch.ops.aten.mm.default(permute_112, arg299_1);  permute_112 = arg299_1 = None
        sum_149: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_354, [0], True, dtype = torch.float32);  view_354 = None
        view_355: "f32[512]" = torch.ops.aten.view.default(sum_149, _shape_param_348);  sum_149 = _shape_param_348 = None
        convert_element_type_258: "bf16[512]" = torch.ops.prims.convert_element_type.default(view_355, torch.bfloat16);  view_355 = None
        view_356: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(mm_83, _shape_param_349);  mm_83 = _shape_param_349 = None
        convert_element_type_259: "f32[512, 2048]" = torch.ops.prims.convert_element_type.default(mm_84, torch.float32);  mm_84 = None
        convert_element_type_260: "f32[512]" = torch.ops.prims.convert_element_type.default(convert_element_type_258, torch.float32);  convert_element_type_258 = None
        convert_element_type_261: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_356, torch.float32);  view_356 = None
        view_357: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(arg300_1, _shape_param_350);  arg300_1 = _shape_param_350 = None
        convert_element_type_262: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_357, torch.float32);  view_357 = None
        mul_265: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_262, 0.7071067811865476)
        erf_10: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_265);  mul_265 = None
        add_55: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_266: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(add_55, 0.5);  add_55 = None
        mul_267: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_262, convert_element_type_262)
        mul_268: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_267, -0.5);  mul_267 = None
        exp_20: "f32[128, 196, 2048]" = torch.ops.aten.exp.default(mul_268);  mul_268 = None
        mul_269: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_20, 0.3989422804014327);  exp_20 = None
        mul_270: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_262, mul_269);  convert_element_type_262 = mul_269 = None
        add_56: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(mul_266, mul_270);  mul_266 = mul_270 = None
        mul_271: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_261, add_56);  convert_element_type_261 = add_56 = None
        convert_element_type_263: "bf16[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(mul_271, torch.bfloat16);  mul_271 = None
        view_358: "bf16[25088, 2048]" = torch.ops.aten.view.default(convert_element_type_263, _shape_param_351);  convert_element_type_263 = _shape_param_351 = None
        mm_85: "bf16[25088, 512]" = torch.ops.aten.mm.default(view_358, arg301_1);  arg301_1 = None
        permute_113: "bf16[2048, 25088]" = torch.ops.aten.permute.default(view_358, [1, 0])
        mm_86: "bf16[2048, 512]" = torch.ops.aten.mm.default(permute_113, arg302_1);  permute_113 = arg302_1 = None
        sum_150: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_358, [0], True, dtype = torch.float32);  view_358 = None
        view_359: "f32[2048]" = torch.ops.aten.view.default(sum_150, _shape_param_352);  sum_150 = _shape_param_352 = None
        convert_element_type_264: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_359, torch.bfloat16);  view_359 = None
        view_360: "bf16[128, 196, 512]" = torch.ops.aten.view.default(mm_85, _shape_param_353);  mm_85 = _shape_param_353 = None
        convert_element_type_265: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(view_360, torch.float32);  view_360 = None
        convert_element_type_266: "f32[2048, 512]" = torch.ops.prims.convert_element_type.default(mm_86, torch.float32);  mm_86 = None
        convert_element_type_267: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_264, torch.float32);  convert_element_type_264 = None
        mul_272: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_265, arg303_1);  arg303_1 = None
        mul_273: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_272, 512)
        sum_151: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_272, [2], True)
        convert_element_type_268: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(arg304_1, torch.float32);  arg304_1 = None
        sub_76: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_268, arg305_1);  convert_element_type_268 = arg305_1 = None
        mul_274: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_76, arg306_1);  sub_76 = None
        mul_275: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_272, mul_274);  mul_272 = None
        sum_152: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_275, [2], True);  mul_275 = None
        mul_276: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_274, sum_152);  sum_152 = None
        sub_77: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(mul_273, sum_151);  mul_273 = sum_151 = None
        sub_78: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(sub_77, mul_276);  sub_77 = mul_276 = None
        div_54: "f32[128, 196, 1]" = torch.ops.aten.div.Tensor(arg306_1, 512);  arg306_1 = None
        mul_277: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(div_54, sub_78);  div_54 = sub_78 = None
        mul_278: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_265, mul_274);  mul_274 = None
        sum_153: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_278, [0, 1]);  mul_278 = None
        sum_154: "f32[512]" = torch.ops.aten.sum.dim_IntList(convert_element_type_265, [0, 1]);  convert_element_type_265 = None
        convert_element_type_269: "bf16[128, 196, 512]" = torch.ops.prims.convert_element_type.default(mul_277, torch.bfloat16);  mul_277 = None
        add_57: "bf16[128, 196, 512]" = torch.ops.aten.add.Tensor(view_353, convert_element_type_269);  view_353 = convert_element_type_269 = None
        view_361: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(add_57, _shape_param_354);  add_57 = _shape_param_354 = None
        convert_element_type_270: "bf16[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(arg307_1, torch.bfloat16);  arg307_1 = None
        div_55: "bf16[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_270, 0.9434782639145851);  convert_element_type_270 = None
        mul_279: "bf16[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_361, div_55);  div_55 = None
        index_26: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(mul_279, [None, None, fmod]);  mul_279 = None
        index_27: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_26, [None, fmod]);  index_26 = None
        view_362: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(index_27, _shape_param_355);  index_27 = _shape_param_355 = None
        permute_114: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_362, [0, 1, 3, 2, 4, 5]);  view_362 = None
        clone_47: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_114, memory_format = torch.contiguous_format);  permute_114 = None
        view_363: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(clone_47, _shape_param_356);  clone_47 = _shape_param_356 = None
        view_364: "bf16[512, 49, 512]" = torch.ops.aten.view.default(view_363, _shape_param_357);  view_363 = _shape_param_357 = None
        view_365: "bf16[25088, 512]" = torch.ops.aten.view.default(view_364, _shape_param_358);  view_364 = _shape_param_358 = None
        mm_87: "bf16[25088, 512]" = torch.ops.aten.mm.default(view_365, arg308_1);  arg308_1 = None
        permute_115: "bf16[512, 25088]" = torch.ops.aten.permute.default(view_365, [1, 0])
        mm_88: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_115, arg309_1);  permute_115 = arg309_1 = None
        sum_155: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_365, [0], True, dtype = torch.float32);  view_365 = None
        view_366: "f32[512]" = torch.ops.aten.view.default(sum_155, _shape_param_359);  sum_155 = _shape_param_359 = None
        convert_element_type_271: "bf16[512]" = torch.ops.prims.convert_element_type.default(view_366, torch.bfloat16);  view_366 = None
        view_367: "bf16[512, 49, 512]" = torch.ops.aten.view.default(mm_87, _shape_param_360);  mm_87 = _shape_param_360 = None
        convert_element_type_272: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_88, torch.float32);  mm_88 = None
        convert_element_type_273: "f32[512]" = torch.ops.prims.convert_element_type.default(convert_element_type_271, torch.float32);  convert_element_type_271 = None
        view_368: "bf16[512, 49, 16, 32]" = torch.ops.aten.view.default(view_367, _shape_param_361);  view_367 = _shape_param_361 = None
        permute_116: "bf16[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_368, [0, 2, 1, 3]);  view_368 = None
        clone_48: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(permute_116, memory_format = torch.contiguous_format);  permute_116 = None
        view_369: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_48, _shape_param_362);  clone_48 = _shape_param_362 = None
        bmm_40: "bf16[8192, 49, 32]" = torch.ops.aten.bmm.default(arg310_1, view_369);  arg310_1 = None
        bmm_41: "bf16[8192, 49, 49]" = torch.ops.aten.bmm.default(view_369, arg311_1);  view_369 = arg311_1 = None
        view_370: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(bmm_40, _shape_param_363);  bmm_40 = _shape_param_363 = None
        view_371: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(bmm_41, _shape_param_364);  bmm_41 = _shape_param_364 = None
        convert_element_type_274: "f32[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(view_371, torch.float32);  view_371 = None
        slice_17: "bf16[8192, 49, 56]" = torch.ops.aten.slice.Tensor(arg312_1, 1, 0, -7);  arg312_1 = None
        slice_18: "bf16[8192, 49, 49]" = torch.ops.aten.slice.Tensor(slice_17, 2, 0, -7);  slice_17 = None
        view_372: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(slice_18, _shape_param_365);  slice_18 = _shape_param_365 = None
        view_373: "i64[2401]" = torch.ops.aten.view.default(arg313_1, [-1]);  arg313_1 = None
        index_28: "f32[2401, 16]" = torch.ops.aten.index.Tensor(arg314_1, [view_373]);  arg314_1 = None
        view_374: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_28, _shape_param_366);  index_28 = _shape_param_366 = None
        permute_117: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_374, [2, 0, 1]);  view_374 = None
        clone_49: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
        unsqueeze_20: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_49, 0);  clone_49 = None
        add_58: "f32[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_372, unsqueeze_20);  view_372 = unsqueeze_20 = None
        view_375: "f32[128, 4, 16, 49, 49]" = torch.ops.aten.view.default(add_58, _shape_param_367);  add_58 = _shape_param_367 = None
        unsqueeze_21: "f32[4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(arg315_1, 1);  arg315_1 = None
        unsqueeze_22: "f32[1, 4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_21, 0);  unsqueeze_21 = None
        add_59: "f32[128, 4, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_375, unsqueeze_22);  view_375 = unsqueeze_22 = None
        view_376: "f32[512, 16, 49, 49]" = torch.ops.aten.view.default(add_59, _shape_param_368);  add_59 = _shape_param_368 = None
        sub_79: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(view_376, arg316_1);  view_376 = arg316_1 = None
        exp_21: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_79);  sub_79 = None
        div_56: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_21, arg317_1);  exp_21 = arg317_1 = None
        mul_280: "f32[512, 16, 49, 49]" = torch.ops.aten.mul.Tensor(convert_element_type_274, div_56);  convert_element_type_274 = None
        sum_156: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_280, [-1], True)
        neg_10: "f32[512, 16, 49, 49]" = torch.ops.aten.neg.default(div_56);  div_56 = None
        fma_10: "f32[512, 16, 49, 49]" = torch.ops.prims.fma.default(neg_10, sum_156, mul_280);  neg_10 = sum_156 = mul_280 = None
        convert_element_type_275: "bf16[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(fma_10, torch.bfloat16)
        sum_157: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_10, [0], True, dtype = torch.float32);  fma_10 = None
        squeeze_10: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_157, 0);  sum_157 = None
        permute_118: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_10, [1, 2, 0]);  squeeze_10 = None
        view_377: "f32[2401, 16]" = torch.ops.aten.view.default(permute_118, _shape_param_369);  permute_118 = _shape_param_369 = None
        index_put_10: "f32[169, 16]" = torch.ops.aten.index_put.default(full_1, [view_373], view_377, True);  view_373 = view_377 = None
        view_378: "bf16[8192, 49, 49]" = torch.ops.aten.view.default(convert_element_type_275, _shape_param_370);  convert_element_type_275 = _shape_param_370 = None
        bmm_42: "bf16[8192, 32, 49]" = torch.ops.aten.bmm.default(arg318_1, view_378);  arg318_1 = None
        bmm_43: "bf16[8192, 49, 32]" = torch.ops.aten.bmm.default(view_378, arg319_1);  view_378 = arg319_1 = None
        view_379: "bf16[512, 16, 32, 49]" = torch.ops.aten.view.default(bmm_42, _shape_param_371);  bmm_42 = _shape_param_371 = None
        view_380: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(bmm_43, _shape_param_372);  bmm_43 = _shape_param_372 = None
        permute_119: "bf16[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_379, [0, 1, 3, 2]);  view_379 = None
        mul_281: "bf16[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_380, 0.1767766952966369);  view_380 = None
        cat_10: "bf16[1536, 16, 49, 32]" = torch.ops.aten.cat.default([mul_281, permute_119, view_370]);  mul_281 = permute_119 = view_370 = None
        view_381: "bf16[3, 512, 16, 49, 32]" = torch.ops.aten.view.default(cat_10, _shape_param_373);  cat_10 = _shape_param_373 = None
        permute_120: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.permute.default(view_381, [1, 3, 0, 2, 4]);  view_381 = None
        clone_50: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_120, memory_format = torch.contiguous_format);  permute_120 = None
        view_382: "bf16[512, 49, 1536]" = torch.ops.aten.view.default(clone_50, _shape_param_374);  clone_50 = _shape_param_374 = None
        view_383: "bf16[25088, 1536]" = torch.ops.aten.view.default(view_382, _shape_param_375);  view_382 = _shape_param_375 = None
        mm_89: "bf16[25088, 512]" = torch.ops.aten.mm.default(view_383, arg320_1);  arg320_1 = None
        permute_121: "bf16[1536, 25088]" = torch.ops.aten.permute.default(view_383, [1, 0])
        mm_90: "bf16[1536, 512]" = torch.ops.aten.mm.default(permute_121, arg321_1);  permute_121 = arg321_1 = None
        sum_158: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_383, [0], True, dtype = torch.float32);  view_383 = None
        view_384: "f32[1536]" = torch.ops.aten.view.default(sum_158, _shape_param_376);  sum_158 = _shape_param_376 = None
        convert_element_type_276: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_384, torch.bfloat16);  view_384 = None
        view_385: "bf16[512, 49, 512]" = torch.ops.aten.view.default(mm_89, _shape_param_377);  mm_89 = _shape_param_377 = None
        convert_element_type_277: "f32[512, 49, 512]" = torch.ops.prims.convert_element_type.default(view_385, torch.float32);  view_385 = None
        convert_element_type_278: "f32[1536, 512]" = torch.ops.prims.convert_element_type.default(mm_90, torch.float32);  mm_90 = None
        convert_element_type_279: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_276, torch.float32);  convert_element_type_276 = None
        view_386: "f32[512, 7, 7, 512]" = torch.ops.aten.view.default(convert_element_type_277, _shape_param_378);  convert_element_type_277 = _shape_param_378 = None
        view_387: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_386, _shape_param_379);  view_386 = _shape_param_379 = None
        permute_122: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_387, [0, 1, 3, 2, 4, 5]);  view_387 = None
        clone_51: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_122, memory_format = torch.contiguous_format);  permute_122 = None
        view_388: "f32[128, 14, 14, 512]" = torch.ops.aten.view.default(clone_51, _shape_param_380);  clone_51 = _shape_param_380 = None
        index_29: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(view_388, [None, None, arg93_1]);  view_388 = None
        index_30: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_29, [None, arg93_1]);  index_29 = None
        mul_282: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(index_30, arg322_1);  arg322_1 = None
        mul_283: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_282, 512)
        sum_159: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_282, [3], True)
        convert_element_type_280: "f32[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(arg323_1, torch.float32);  arg323_1 = None
        sub_80: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_280, arg324_1);  convert_element_type_280 = arg324_1 = None
        mul_284: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_80, arg325_1);  sub_80 = None
        mul_285: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_282, mul_284);  mul_282 = None
        sum_160: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_285, [3], True);  mul_285 = None
        mul_286: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_284, sum_160);  sum_160 = None
        sub_81: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_283, sum_159);  mul_283 = sum_159 = None
        sub_82: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_81, mul_286);  sub_81 = mul_286 = None
        div_57: "f32[128, 14, 14, 1]" = torch.ops.aten.div.Tensor(arg325_1, 512);  arg325_1 = None
        mul_287: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_57, sub_82);  div_57 = sub_82 = None
        mul_288: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(index_30, mul_284);  mul_284 = None
        sum_161: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_288, [0, 1, 2]);  mul_288 = None
        sum_162: "f32[512]" = torch.ops.aten.sum.dim_IntList(index_30, [0, 1, 2]);  index_30 = None
        convert_element_type_281: "bf16[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(mul_287, torch.bfloat16);  mul_287 = None
        add_60: "bf16[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_361, convert_element_type_281);  view_361 = convert_element_type_281 = None
        view_389: "bf16[128, 196, 512]" = torch.ops.aten.view.default(add_60, _shape_param_381);  add_60 = _shape_param_381 = None
        convert_element_type_282: "bf16[128, 1, 1]" = torch.ops.prims.convert_element_type.default(arg326_1, torch.bfloat16);  arg326_1 = None
        div_58: "bf16[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_282, 0.947826087474823);  convert_element_type_282 = None
        mul_289: "bf16[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_389, div_58);  div_58 = None
        view_390: "bf16[25088, 512]" = torch.ops.aten.view.default(mul_289, _shape_param_382);  mul_289 = _shape_param_382 = None
        mm_91: "bf16[25088, 2048]" = torch.ops.aten.mm.default(view_390, arg327_1);  arg327_1 = None
        permute_123: "bf16[512, 25088]" = torch.ops.aten.permute.default(view_390, [1, 0])
        mm_92: "bf16[512, 2048]" = torch.ops.aten.mm.default(permute_123, arg328_1);  permute_123 = arg328_1 = None
        sum_163: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_390, [0], True, dtype = torch.float32);  view_390 = None
        view_391: "f32[512]" = torch.ops.aten.view.default(sum_163, _shape_param_383);  sum_163 = _shape_param_383 = None
        convert_element_type_283: "bf16[512]" = torch.ops.prims.convert_element_type.default(view_391, torch.bfloat16);  view_391 = None
        view_392: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(mm_91, _shape_param_384);  mm_91 = _shape_param_384 = None
        convert_element_type_284: "f32[512, 2048]" = torch.ops.prims.convert_element_type.default(mm_92, torch.float32);  mm_92 = None
        convert_element_type_285: "f32[512]" = torch.ops.prims.convert_element_type.default(convert_element_type_283, torch.float32);  convert_element_type_283 = None
        convert_element_type_286: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_392, torch.float32);  view_392 = None
        view_393: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(arg329_1, _shape_param_385);  arg329_1 = _shape_param_385 = None
        convert_element_type_287: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_393, torch.float32);  view_393 = None
        mul_290: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_287, 0.7071067811865476)
        erf_11: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_290);  mul_290 = None
        add_61: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_291: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(add_61, 0.5);  add_61 = None
        mul_292: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_287, convert_element_type_287)
        mul_293: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_292, -0.5);  mul_292 = None
        exp_22: "f32[128, 196, 2048]" = torch.ops.aten.exp.default(mul_293);  mul_293 = None
        mul_294: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_22, 0.3989422804014327);  exp_22 = None
        mul_295: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_287, mul_294);  convert_element_type_287 = mul_294 = None
        add_62: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(mul_291, mul_295);  mul_291 = mul_295 = None
        mul_296: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_286, add_62);  convert_element_type_286 = add_62 = None
        convert_element_type_288: "bf16[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(mul_296, torch.bfloat16);  mul_296 = None
        view_394: "bf16[25088, 2048]" = torch.ops.aten.view.default(convert_element_type_288, _shape_param_386);  convert_element_type_288 = _shape_param_386 = None
        mm_93: "bf16[25088, 512]" = torch.ops.aten.mm.default(view_394, arg330_1);  arg330_1 = None
        permute_124: "bf16[2048, 25088]" = torch.ops.aten.permute.default(view_394, [1, 0])
        mm_94: "bf16[2048, 512]" = torch.ops.aten.mm.default(permute_124, arg331_1);  permute_124 = arg331_1 = None
        sum_164: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_394, [0], True, dtype = torch.float32);  view_394 = None
        view_395: "f32[2048]" = torch.ops.aten.view.default(sum_164, _shape_param_387);  sum_164 = _shape_param_387 = None
        convert_element_type_289: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_395, torch.bfloat16);  view_395 = None
        view_396: "bf16[128, 196, 512]" = torch.ops.aten.view.default(mm_93, _shape_param_388);  mm_93 = _shape_param_388 = None
        convert_element_type_290: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(view_396, torch.float32);  view_396 = None
        convert_element_type_291: "f32[2048, 512]" = torch.ops.prims.convert_element_type.default(mm_94, torch.float32);  mm_94 = None
        convert_element_type_292: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_289, torch.float32);  convert_element_type_289 = None
        mul_297: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_290, arg332_1);  arg332_1 = None
        mul_298: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_297, 512)
        sum_165: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_297, [2], True)
        convert_element_type_293: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(arg333_1, torch.float32);  arg333_1 = None
        sub_83: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_293, arg334_1);  convert_element_type_293 = arg334_1 = None
        mul_299: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_83, arg335_1);  sub_83 = None
        mul_300: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_297, mul_299);  mul_297 = None
        sum_166: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_300, [2], True);  mul_300 = None
        mul_301: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_299, sum_166);  sum_166 = None
        sub_84: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(mul_298, sum_165);  mul_298 = sum_165 = None
        sub_85: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(sub_84, mul_301);  sub_84 = mul_301 = None
        div_59: "f32[128, 196, 1]" = torch.ops.aten.div.Tensor(arg335_1, 512);  arg335_1 = None
        mul_302: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(div_59, sub_85);  div_59 = sub_85 = None
        mul_303: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_290, mul_299);  mul_299 = None
        sum_167: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_303, [0, 1]);  mul_303 = None
        sum_168: "f32[512]" = torch.ops.aten.sum.dim_IntList(convert_element_type_290, [0, 1]);  convert_element_type_290 = None
        convert_element_type_294: "bf16[128, 196, 512]" = torch.ops.prims.convert_element_type.default(mul_302, torch.bfloat16);  mul_302 = None
        add_63: "bf16[128, 196, 512]" = torch.ops.aten.add.Tensor(view_389, convert_element_type_294);  view_389 = convert_element_type_294 = None
        view_397: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(add_63, _shape_param_389);  add_63 = _shape_param_389 = None
        convert_element_type_295: "bf16[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(arg336_1, torch.bfloat16);  arg336_1 = None
        div_60: "bf16[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_295, 0.947826087474823);  convert_element_type_295 = None
        mul_304: "bf16[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_397, div_60);  div_60 = None
        view_398: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(mul_304, _shape_param_390);  mul_304 = _shape_param_390 = None
        permute_125: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_398, [0, 1, 3, 2, 4, 5]);  view_398 = None
        clone_52: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_125, memory_format = torch.contiguous_format);  permute_125 = None
        view_399: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(clone_52, _shape_param_391);  clone_52 = _shape_param_391 = None
        view_400: "bf16[512, 49, 512]" = torch.ops.aten.view.default(view_399, _shape_param_392);  view_399 = _shape_param_392 = None
        view_401: "bf16[25088, 512]" = torch.ops.aten.view.default(view_400, _shape_param_393);  view_400 = _shape_param_393 = None
        mm_95: "bf16[25088, 512]" = torch.ops.aten.mm.default(view_401, arg337_1);  arg337_1 = None
        permute_126: "bf16[512, 25088]" = torch.ops.aten.permute.default(view_401, [1, 0])
        mm_96: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_126, arg338_1);  permute_126 = arg338_1 = None
        sum_169: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_401, [0], True, dtype = torch.float32);  view_401 = None
        view_402: "f32[512]" = torch.ops.aten.view.default(sum_169, _shape_param_394);  sum_169 = _shape_param_394 = None
        convert_element_type_296: "bf16[512]" = torch.ops.prims.convert_element_type.default(view_402, torch.bfloat16);  view_402 = None
        view_403: "bf16[512, 49, 512]" = torch.ops.aten.view.default(mm_95, _shape_param_395);  mm_95 = _shape_param_395 = None
        convert_element_type_297: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_96, torch.float32);  mm_96 = None
        convert_element_type_298: "f32[512]" = torch.ops.prims.convert_element_type.default(convert_element_type_296, torch.float32);  convert_element_type_296 = None
        view_404: "bf16[512, 49, 16, 32]" = torch.ops.aten.view.default(view_403, _shape_param_396);  view_403 = _shape_param_396 = None
        permute_127: "bf16[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_404, [0, 2, 1, 3]);  view_404 = None
        clone_53: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(permute_127, memory_format = torch.contiguous_format);  permute_127 = None
        view_405: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_53, _shape_param_397);  clone_53 = _shape_param_397 = None
        bmm_44: "bf16[8192, 49, 32]" = torch.ops.aten.bmm.default(arg339_1, view_405);  arg339_1 = None
        bmm_45: "bf16[8192, 49, 49]" = torch.ops.aten.bmm.default(view_405, arg340_1);  view_405 = arg340_1 = None
        view_406: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(bmm_44, _shape_param_398);  bmm_44 = _shape_param_398 = None
        view_407: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(bmm_45, _shape_param_399);  bmm_45 = _shape_param_399 = None
        convert_element_type_299: "f32[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(view_407, torch.float32);  view_407 = None
        slice_19: "bf16[8192, 49, 56]" = torch.ops.aten.slice.Tensor(arg341_1, 1, 0, -7);  arg341_1 = None
        slice_20: "bf16[8192, 49, 49]" = torch.ops.aten.slice.Tensor(slice_19, 2, 0, -7);  slice_19 = None
        view_408: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(slice_20, _shape_param_400);  slice_20 = _shape_param_400 = None
        view_409: "i64[2401]" = torch.ops.aten.view.default(arg342_1, [-1]);  arg342_1 = None
        index_31: "f32[2401, 16]" = torch.ops.aten.index.Tensor(arg343_1, [view_409]);  arg343_1 = None
        view_410: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_31, _shape_param_401);  index_31 = _shape_param_401 = None
        permute_128: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_410, [2, 0, 1]);  view_410 = None
        clone_54: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
        unsqueeze_23: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_54, 0);  clone_54 = None
        add_64: "f32[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_408, unsqueeze_23);  view_408 = unsqueeze_23 = None
        sub_86: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(add_64, arg344_1);  add_64 = arg344_1 = None
        exp_23: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_86);  sub_86 = None
        div_61: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_23, arg345_1);  exp_23 = arg345_1 = None
        mul_305: "f32[512, 16, 49, 49]" = torch.ops.aten.mul.Tensor(convert_element_type_299, div_61);  convert_element_type_299 = None
        sum_170: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_305, [-1], True)
        neg_11: "f32[512, 16, 49, 49]" = torch.ops.aten.neg.default(div_61);  div_61 = None
        fma_11: "f32[512, 16, 49, 49]" = torch.ops.prims.fma.default(neg_11, sum_170, mul_305);  neg_11 = sum_170 = mul_305 = None
        convert_element_type_300: "bf16[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(fma_11, torch.bfloat16)
        sum_171: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_11, [0], True, dtype = torch.float32);  fma_11 = None
        squeeze_11: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_171, 0);  sum_171 = None
        permute_129: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_11, [1, 2, 0]);  squeeze_11 = None
        view_411: "f32[2401, 16]" = torch.ops.aten.view.default(permute_129, _shape_param_402);  permute_129 = _shape_param_402 = None
        index_put_11: "f32[169, 16]" = torch.ops.aten.index_put.default(full_1, [view_409], view_411, True);  view_409 = view_411 = None
        view_412: "bf16[8192, 49, 49]" = torch.ops.aten.view.default(convert_element_type_300, _shape_param_403);  convert_element_type_300 = _shape_param_403 = None
        bmm_46: "bf16[8192, 32, 49]" = torch.ops.aten.bmm.default(arg346_1, view_412);  arg346_1 = None
        bmm_47: "bf16[8192, 49, 32]" = torch.ops.aten.bmm.default(view_412, arg347_1);  view_412 = arg347_1 = None
        view_413: "bf16[512, 16, 32, 49]" = torch.ops.aten.view.default(bmm_46, _shape_param_404);  bmm_46 = _shape_param_404 = None
        view_414: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(bmm_47, _shape_param_405);  bmm_47 = _shape_param_405 = None
        permute_130: "bf16[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_413, [0, 1, 3, 2]);  view_413 = None
        mul_306: "bf16[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_414, 0.1767766952966369);  view_414 = None
        cat_11: "bf16[1536, 16, 49, 32]" = torch.ops.aten.cat.default([mul_306, permute_130, view_406]);  mul_306 = permute_130 = view_406 = None
        view_415: "bf16[3, 512, 16, 49, 32]" = torch.ops.aten.view.default(cat_11, _shape_param_406);  cat_11 = _shape_param_406 = None
        permute_131: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.permute.default(view_415, [1, 3, 0, 2, 4]);  view_415 = None
        clone_55: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_131, memory_format = torch.contiguous_format);  permute_131 = None
        view_416: "bf16[512, 49, 1536]" = torch.ops.aten.view.default(clone_55, _shape_param_407);  clone_55 = _shape_param_407 = None
        view_417: "bf16[25088, 1536]" = torch.ops.aten.view.default(view_416, _shape_param_408);  view_416 = _shape_param_408 = None
        mm_97: "bf16[25088, 512]" = torch.ops.aten.mm.default(view_417, arg348_1);  arg348_1 = None
        permute_132: "bf16[1536, 25088]" = torch.ops.aten.permute.default(view_417, [1, 0])
        mm_98: "bf16[1536, 512]" = torch.ops.aten.mm.default(permute_132, arg349_1);  permute_132 = arg349_1 = None
        sum_172: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_417, [0], True, dtype = torch.float32);  view_417 = None
        view_418: "f32[1536]" = torch.ops.aten.view.default(sum_172, _shape_param_409);  sum_172 = _shape_param_409 = None
        convert_element_type_301: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_418, torch.bfloat16);  view_418 = None
        view_419: "bf16[512, 49, 512]" = torch.ops.aten.view.default(mm_97, _shape_param_410);  mm_97 = _shape_param_410 = None
        convert_element_type_302: "f32[512, 49, 512]" = torch.ops.prims.convert_element_type.default(view_419, torch.float32);  view_419 = None
        convert_element_type_303: "f32[1536, 512]" = torch.ops.prims.convert_element_type.default(mm_98, torch.float32);  mm_98 = None
        convert_element_type_304: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_301, torch.float32);  convert_element_type_301 = None
        view_420: "f32[512, 7, 7, 512]" = torch.ops.aten.view.default(convert_element_type_302, _shape_param_411);  convert_element_type_302 = _shape_param_411 = None
        view_421: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_420, _shape_param_412);  view_420 = _shape_param_412 = None
        permute_133: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_421, [0, 1, 3, 2, 4, 5]);  view_421 = None
        clone_56: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_133, memory_format = torch.contiguous_format);  permute_133 = None
        view_422: "f32[128, 14, 14, 512]" = torch.ops.aten.view.default(clone_56, _shape_param_413);  clone_56 = _shape_param_413 = None
        mul_307: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_422, arg350_1);  arg350_1 = None
        mul_308: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_307, 512)
        sum_173: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_307, [3], True)
        convert_element_type_305: "f32[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(arg351_1, torch.float32);  arg351_1 = None
        sub_87: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_305, arg352_1);  convert_element_type_305 = arg352_1 = None
        mul_309: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_87, arg353_1);  sub_87 = None
        mul_310: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_307, mul_309);  mul_307 = None
        sum_174: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_310, [3], True);  mul_310 = None
        mul_311: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_309, sum_174);  sum_174 = None
        sub_88: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_308, sum_173);  mul_308 = sum_173 = None
        sub_89: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_88, mul_311);  sub_88 = mul_311 = None
        div_62: "f32[128, 14, 14, 1]" = torch.ops.aten.div.Tensor(arg353_1, 512);  arg353_1 = None
        mul_312: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_62, sub_89);  div_62 = sub_89 = None
        mul_313: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_422, mul_309);  mul_309 = None
        sum_175: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_313, [0, 1, 2]);  mul_313 = None
        sum_176: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_422, [0, 1, 2]);  view_422 = None
        convert_element_type_306: "bf16[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(mul_312, torch.bfloat16);  mul_312 = None
        add_65: "bf16[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_397, convert_element_type_306);  view_397 = convert_element_type_306 = None
        view_423: "bf16[128, 196, 512]" = torch.ops.aten.view.default(add_65, _shape_param_414);  add_65 = _shape_param_414 = None
        convert_element_type_307: "bf16[128, 1, 1]" = torch.ops.prims.convert_element_type.default(arg354_1, torch.bfloat16);  arg354_1 = None
        div_63: "bf16[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_307, 0.9521739110350609);  convert_element_type_307 = None
        mul_314: "bf16[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_423, div_63);  div_63 = None
        view_424: "bf16[25088, 512]" = torch.ops.aten.view.default(mul_314, _shape_param_415);  mul_314 = _shape_param_415 = None
        mm_99: "bf16[25088, 2048]" = torch.ops.aten.mm.default(view_424, arg355_1);  arg355_1 = None
        permute_134: "bf16[512, 25088]" = torch.ops.aten.permute.default(view_424, [1, 0])
        mm_100: "bf16[512, 2048]" = torch.ops.aten.mm.default(permute_134, arg356_1);  permute_134 = arg356_1 = None
        sum_177: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_424, [0], True, dtype = torch.float32);  view_424 = None
        view_425: "f32[512]" = torch.ops.aten.view.default(sum_177, _shape_param_416);  sum_177 = _shape_param_416 = None
        convert_element_type_308: "bf16[512]" = torch.ops.prims.convert_element_type.default(view_425, torch.bfloat16);  view_425 = None
        view_426: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(mm_99, _shape_param_417);  mm_99 = _shape_param_417 = None
        convert_element_type_309: "f32[512, 2048]" = torch.ops.prims.convert_element_type.default(mm_100, torch.float32);  mm_100 = None
        convert_element_type_310: "f32[512]" = torch.ops.prims.convert_element_type.default(convert_element_type_308, torch.float32);  convert_element_type_308 = None
        convert_element_type_311: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_426, torch.float32);  view_426 = None
        view_427: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(arg357_1, _shape_param_418);  arg357_1 = _shape_param_418 = None
        convert_element_type_312: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_427, torch.float32);  view_427 = None
        mul_315: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_312, 0.7071067811865476)
        erf_12: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_315);  mul_315 = None
        add_66: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_316: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(add_66, 0.5);  add_66 = None
        mul_317: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_312, convert_element_type_312)
        mul_318: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_317, -0.5);  mul_317 = None
        exp_24: "f32[128, 196, 2048]" = torch.ops.aten.exp.default(mul_318);  mul_318 = None
        mul_319: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_24, 0.3989422804014327);  exp_24 = None
        mul_320: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_312, mul_319);  convert_element_type_312 = mul_319 = None
        add_67: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(mul_316, mul_320);  mul_316 = mul_320 = None
        mul_321: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_311, add_67);  convert_element_type_311 = add_67 = None
        convert_element_type_313: "bf16[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(mul_321, torch.bfloat16);  mul_321 = None
        view_428: "bf16[25088, 2048]" = torch.ops.aten.view.default(convert_element_type_313, _shape_param_419);  convert_element_type_313 = _shape_param_419 = None
        mm_101: "bf16[25088, 512]" = torch.ops.aten.mm.default(view_428, arg358_1);  arg358_1 = None
        permute_135: "bf16[2048, 25088]" = torch.ops.aten.permute.default(view_428, [1, 0])
        mm_102: "bf16[2048, 512]" = torch.ops.aten.mm.default(permute_135, arg359_1);  permute_135 = arg359_1 = None
        sum_178: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_428, [0], True, dtype = torch.float32);  view_428 = None
        view_429: "f32[2048]" = torch.ops.aten.view.default(sum_178, _shape_param_420);  sum_178 = _shape_param_420 = None
        convert_element_type_314: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_429, torch.bfloat16);  view_429 = None
        view_430: "bf16[128, 196, 512]" = torch.ops.aten.view.default(mm_101, _shape_param_421);  mm_101 = _shape_param_421 = None
        convert_element_type_315: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(view_430, torch.float32);  view_430 = None
        convert_element_type_316: "f32[2048, 512]" = torch.ops.prims.convert_element_type.default(mm_102, torch.float32);  mm_102 = None
        convert_element_type_317: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_314, torch.float32);  convert_element_type_314 = None
        mul_322: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_315, arg360_1);  arg360_1 = None
        mul_323: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_322, 512)
        sum_179: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_322, [2], True)
        convert_element_type_318: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(arg361_1, torch.float32);  arg361_1 = None
        sub_90: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_318, arg362_1);  convert_element_type_318 = arg362_1 = None
        mul_324: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_90, arg363_1);  sub_90 = None
        mul_325: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_322, mul_324);  mul_322 = None
        sum_180: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_325, [2], True);  mul_325 = None
        mul_326: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_324, sum_180);  sum_180 = None
        sub_91: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(mul_323, sum_179);  mul_323 = sum_179 = None
        sub_92: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(sub_91, mul_326);  sub_91 = mul_326 = None
        div_64: "f32[128, 196, 1]" = torch.ops.aten.div.Tensor(arg363_1, 512);  arg363_1 = None
        mul_327: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(div_64, sub_92);  div_64 = sub_92 = None
        mul_328: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_315, mul_324);  mul_324 = None
        sum_181: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_328, [0, 1]);  mul_328 = None
        sum_182: "f32[512]" = torch.ops.aten.sum.dim_IntList(convert_element_type_315, [0, 1]);  convert_element_type_315 = None
        convert_element_type_319: "bf16[128, 196, 512]" = torch.ops.prims.convert_element_type.default(mul_327, torch.bfloat16);  mul_327 = None
        add_68: "bf16[128, 196, 512]" = torch.ops.aten.add.Tensor(view_423, convert_element_type_319);  view_423 = convert_element_type_319 = None
        view_431: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(add_68, _shape_param_422);  add_68 = _shape_param_422 = None
        convert_element_type_320: "bf16[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(arg364_1, torch.bfloat16);  arg364_1 = None
        div_65: "bf16[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_320, 0.9521739110350609);  convert_element_type_320 = None
        mul_329: "bf16[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_431, div_65);  div_65 = None
        index_32: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(mul_329, [None, None, fmod]);  mul_329 = None
        index_33: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_32, [None, fmod]);  index_32 = None
        view_432: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(index_33, _shape_param_423);  index_33 = _shape_param_423 = None
        permute_136: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_432, [0, 1, 3, 2, 4, 5]);  view_432 = None
        clone_57: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_136, memory_format = torch.contiguous_format);  permute_136 = None
        view_433: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(clone_57, _shape_param_424);  clone_57 = _shape_param_424 = None
        view_434: "bf16[512, 49, 512]" = torch.ops.aten.view.default(view_433, _shape_param_425);  view_433 = _shape_param_425 = None
        view_435: "bf16[25088, 512]" = torch.ops.aten.view.default(view_434, _shape_param_426);  view_434 = _shape_param_426 = None
        mm_103: "bf16[25088, 512]" = torch.ops.aten.mm.default(view_435, arg365_1);  arg365_1 = None
        permute_137: "bf16[512, 25088]" = torch.ops.aten.permute.default(view_435, [1, 0])
        mm_104: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_137, arg366_1);  permute_137 = arg366_1 = None
        sum_183: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_435, [0], True, dtype = torch.float32);  view_435 = None
        view_436: "f32[512]" = torch.ops.aten.view.default(sum_183, _shape_param_427);  sum_183 = _shape_param_427 = None
        convert_element_type_321: "bf16[512]" = torch.ops.prims.convert_element_type.default(view_436, torch.bfloat16);  view_436 = None
        view_437: "bf16[512, 49, 512]" = torch.ops.aten.view.default(mm_103, _shape_param_428);  mm_103 = _shape_param_428 = None
        convert_element_type_322: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_104, torch.float32);  mm_104 = None
        convert_element_type_323: "f32[512]" = torch.ops.prims.convert_element_type.default(convert_element_type_321, torch.float32);  convert_element_type_321 = None
        view_438: "bf16[512, 49, 16, 32]" = torch.ops.aten.view.default(view_437, _shape_param_429);  view_437 = _shape_param_429 = None
        permute_138: "bf16[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_438, [0, 2, 1, 3]);  view_438 = None
        clone_58: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(permute_138, memory_format = torch.contiguous_format);  permute_138 = None
        view_439: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_58, _shape_param_430);  clone_58 = _shape_param_430 = None
        bmm_48: "bf16[8192, 49, 32]" = torch.ops.aten.bmm.default(arg367_1, view_439);  arg367_1 = None
        bmm_49: "bf16[8192, 49, 49]" = torch.ops.aten.bmm.default(view_439, arg368_1);  view_439 = arg368_1 = None
        view_440: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(bmm_48, _shape_param_431);  bmm_48 = _shape_param_431 = None
        view_441: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(bmm_49, _shape_param_432);  bmm_49 = _shape_param_432 = None
        convert_element_type_324: "f32[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(view_441, torch.float32);  view_441 = None
        slice_21: "bf16[8192, 49, 56]" = torch.ops.aten.slice.Tensor(arg369_1, 1, 0, -7);  arg369_1 = None
        slice_22: "bf16[8192, 49, 49]" = torch.ops.aten.slice.Tensor(slice_21, 2, 0, -7);  slice_21 = None
        view_442: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(slice_22, _shape_param_433);  slice_22 = _shape_param_433 = None
        view_443: "i64[2401]" = torch.ops.aten.view.default(arg370_1, [-1]);  arg370_1 = None
        index_34: "f32[2401, 16]" = torch.ops.aten.index.Tensor(arg371_1, [view_443]);  arg371_1 = None
        view_444: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_34, _shape_param_434);  index_34 = _shape_param_434 = None
        permute_139: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_444, [2, 0, 1]);  view_444 = None
        clone_59: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_139, memory_format = torch.contiguous_format);  permute_139 = None
        unsqueeze_24: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_59, 0);  clone_59 = None
        add_69: "f32[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_442, unsqueeze_24);  view_442 = unsqueeze_24 = None
        view_445: "f32[128, 4, 16, 49, 49]" = torch.ops.aten.view.default(add_69, _shape_param_435);  add_69 = _shape_param_435 = None
        unsqueeze_25: "f32[4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(arg372_1, 1);  arg372_1 = None
        unsqueeze_26: "f32[1, 4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_25, 0);  unsqueeze_25 = None
        add_70: "f32[128, 4, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_445, unsqueeze_26);  view_445 = unsqueeze_26 = None
        view_446: "f32[512, 16, 49, 49]" = torch.ops.aten.view.default(add_70, _shape_param_436);  add_70 = _shape_param_436 = None
        sub_93: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(view_446, arg373_1);  view_446 = arg373_1 = None
        exp_25: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_93);  sub_93 = None
        div_66: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_25, arg374_1);  exp_25 = arg374_1 = None
        mul_330: "f32[512, 16, 49, 49]" = torch.ops.aten.mul.Tensor(convert_element_type_324, div_66);  convert_element_type_324 = None
        sum_184: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_330, [-1], True)
        neg_12: "f32[512, 16, 49, 49]" = torch.ops.aten.neg.default(div_66);  div_66 = None
        fma_12: "f32[512, 16, 49, 49]" = torch.ops.prims.fma.default(neg_12, sum_184, mul_330);  neg_12 = sum_184 = mul_330 = None
        convert_element_type_325: "bf16[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(fma_12, torch.bfloat16)
        sum_185: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_12, [0], True, dtype = torch.float32);  fma_12 = None
        squeeze_12: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_185, 0);  sum_185 = None
        permute_140: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_12, [1, 2, 0]);  squeeze_12 = None
        view_447: "f32[2401, 16]" = torch.ops.aten.view.default(permute_140, _shape_param_437);  permute_140 = _shape_param_437 = None
        index_put_12: "f32[169, 16]" = torch.ops.aten.index_put.default(full_1, [view_443], view_447, True);  view_443 = view_447 = None
        view_448: "bf16[8192, 49, 49]" = torch.ops.aten.view.default(convert_element_type_325, _shape_param_438);  convert_element_type_325 = _shape_param_438 = None
        bmm_50: "bf16[8192, 32, 49]" = torch.ops.aten.bmm.default(arg375_1, view_448);  arg375_1 = None
        bmm_51: "bf16[8192, 49, 32]" = torch.ops.aten.bmm.default(view_448, arg376_1);  view_448 = arg376_1 = None
        view_449: "bf16[512, 16, 32, 49]" = torch.ops.aten.view.default(bmm_50, _shape_param_439);  bmm_50 = _shape_param_439 = None
        view_450: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(bmm_51, _shape_param_440);  bmm_51 = _shape_param_440 = None
        permute_141: "bf16[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_449, [0, 1, 3, 2]);  view_449 = None
        mul_331: "bf16[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_450, 0.1767766952966369);  view_450 = None
        cat_12: "bf16[1536, 16, 49, 32]" = torch.ops.aten.cat.default([mul_331, permute_141, view_440]);  mul_331 = permute_141 = view_440 = None
        view_451: "bf16[3, 512, 16, 49, 32]" = torch.ops.aten.view.default(cat_12, _shape_param_441);  cat_12 = _shape_param_441 = None
        permute_142: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.permute.default(view_451, [1, 3, 0, 2, 4]);  view_451 = None
        clone_60: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_142, memory_format = torch.contiguous_format);  permute_142 = None
        view_452: "bf16[512, 49, 1536]" = torch.ops.aten.view.default(clone_60, _shape_param_442);  clone_60 = _shape_param_442 = None
        view_453: "bf16[25088, 1536]" = torch.ops.aten.view.default(view_452, _shape_param_443);  view_452 = _shape_param_443 = None
        mm_105: "bf16[25088, 512]" = torch.ops.aten.mm.default(view_453, arg377_1);  arg377_1 = None
        permute_143: "bf16[1536, 25088]" = torch.ops.aten.permute.default(view_453, [1, 0])
        mm_106: "bf16[1536, 512]" = torch.ops.aten.mm.default(permute_143, arg378_1);  permute_143 = arg378_1 = None
        sum_186: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_453, [0], True, dtype = torch.float32);  view_453 = None
        view_454: "f32[1536]" = torch.ops.aten.view.default(sum_186, _shape_param_444);  sum_186 = _shape_param_444 = None
        convert_element_type_326: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_454, torch.bfloat16);  view_454 = None
        view_455: "bf16[512, 49, 512]" = torch.ops.aten.view.default(mm_105, _shape_param_445);  mm_105 = _shape_param_445 = None
        convert_element_type_327: "f32[512, 49, 512]" = torch.ops.prims.convert_element_type.default(view_455, torch.float32);  view_455 = None
        convert_element_type_328: "f32[1536, 512]" = torch.ops.prims.convert_element_type.default(mm_106, torch.float32);  mm_106 = None
        convert_element_type_329: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_326, torch.float32);  convert_element_type_326 = None
        view_456: "f32[512, 7, 7, 512]" = torch.ops.aten.view.default(convert_element_type_327, _shape_param_446);  convert_element_type_327 = _shape_param_446 = None
        view_457: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_456, _shape_param_447);  view_456 = _shape_param_447 = None
        permute_144: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_457, [0, 1, 3, 2, 4, 5]);  view_457 = None
        clone_61: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_144, memory_format = torch.contiguous_format);  permute_144 = None
        view_458: "f32[128, 14, 14, 512]" = torch.ops.aten.view.default(clone_61, _shape_param_448);  clone_61 = _shape_param_448 = None
        index_35: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(view_458, [None, None, arg93_1]);  view_458 = None
        index_36: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_35, [None, arg93_1]);  index_35 = None
        mul_332: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(index_36, arg379_1);  arg379_1 = None
        mul_333: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_332, 512)
        sum_187: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_332, [3], True)
        convert_element_type_330: "f32[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(arg380_1, torch.float32);  arg380_1 = None
        sub_94: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_330, arg381_1);  convert_element_type_330 = arg381_1 = None
        mul_334: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_94, arg382_1);  sub_94 = None
        mul_335: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_332, mul_334);  mul_332 = None
        sum_188: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_335, [3], True);  mul_335 = None
        mul_336: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_334, sum_188);  sum_188 = None
        sub_95: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_333, sum_187);  mul_333 = sum_187 = None
        sub_96: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_95, mul_336);  sub_95 = mul_336 = None
        div_67: "f32[128, 14, 14, 1]" = torch.ops.aten.div.Tensor(arg382_1, 512);  arg382_1 = None
        mul_337: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_67, sub_96);  div_67 = sub_96 = None
        mul_338: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(index_36, mul_334);  mul_334 = None
        sum_189: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_338, [0, 1, 2]);  mul_338 = None
        sum_190: "f32[512]" = torch.ops.aten.sum.dim_IntList(index_36, [0, 1, 2]);  index_36 = None
        convert_element_type_331: "bf16[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(mul_337, torch.bfloat16);  mul_337 = None
        add_71: "bf16[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_431, convert_element_type_331);  view_431 = convert_element_type_331 = None
        view_459: "bf16[128, 196, 512]" = torch.ops.aten.view.default(add_71, _shape_param_449);  add_71 = _shape_param_449 = None
        convert_element_type_332: "bf16[128, 1, 1]" = torch.ops.prims.convert_element_type.default(arg383_1, torch.bfloat16);  arg383_1 = None
        div_68: "bf16[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_332, 0.9565217345952988);  convert_element_type_332 = None
        mul_339: "bf16[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_459, div_68);  div_68 = None
        view_460: "bf16[25088, 512]" = torch.ops.aten.view.default(mul_339, _shape_param_450);  mul_339 = _shape_param_450 = None
        mm_107: "bf16[25088, 2048]" = torch.ops.aten.mm.default(view_460, arg384_1);  arg384_1 = None
        permute_145: "bf16[512, 25088]" = torch.ops.aten.permute.default(view_460, [1, 0])
        mm_108: "bf16[512, 2048]" = torch.ops.aten.mm.default(permute_145, arg385_1);  permute_145 = arg385_1 = None
        sum_191: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_460, [0], True, dtype = torch.float32);  view_460 = None
        view_461: "f32[512]" = torch.ops.aten.view.default(sum_191, _shape_param_451);  sum_191 = _shape_param_451 = None
        convert_element_type_333: "bf16[512]" = torch.ops.prims.convert_element_type.default(view_461, torch.bfloat16);  view_461 = None
        view_462: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(mm_107, _shape_param_452);  mm_107 = _shape_param_452 = None
        convert_element_type_334: "f32[512, 2048]" = torch.ops.prims.convert_element_type.default(mm_108, torch.float32);  mm_108 = None
        convert_element_type_335: "f32[512]" = torch.ops.prims.convert_element_type.default(convert_element_type_333, torch.float32);  convert_element_type_333 = None
        convert_element_type_336: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_462, torch.float32);  view_462 = None
        view_463: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(arg386_1, _shape_param_453);  arg386_1 = _shape_param_453 = None
        convert_element_type_337: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_463, torch.float32);  view_463 = None
        mul_340: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_337, 0.7071067811865476)
        erf_13: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_340);  mul_340 = None
        add_72: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
        mul_341: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(add_72, 0.5);  add_72 = None
        mul_342: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_337, convert_element_type_337)
        mul_343: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_342, -0.5);  mul_342 = None
        exp_26: "f32[128, 196, 2048]" = torch.ops.aten.exp.default(mul_343);  mul_343 = None
        mul_344: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_26, 0.3989422804014327);  exp_26 = None
        mul_345: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_337, mul_344);  convert_element_type_337 = mul_344 = None
        add_73: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(mul_341, mul_345);  mul_341 = mul_345 = None
        mul_346: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_336, add_73);  convert_element_type_336 = add_73 = None
        convert_element_type_338: "bf16[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(mul_346, torch.bfloat16);  mul_346 = None
        view_464: "bf16[25088, 2048]" = torch.ops.aten.view.default(convert_element_type_338, _shape_param_454);  convert_element_type_338 = _shape_param_454 = None
        mm_109: "bf16[25088, 512]" = torch.ops.aten.mm.default(view_464, arg387_1);  arg387_1 = None
        permute_146: "bf16[2048, 25088]" = torch.ops.aten.permute.default(view_464, [1, 0])
        mm_110: "bf16[2048, 512]" = torch.ops.aten.mm.default(permute_146, arg388_1);  permute_146 = arg388_1 = None
        sum_192: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_464, [0], True, dtype = torch.float32);  view_464 = None
        view_465: "f32[2048]" = torch.ops.aten.view.default(sum_192, _shape_param_455);  sum_192 = _shape_param_455 = None
        convert_element_type_339: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_465, torch.bfloat16);  view_465 = None
        view_466: "bf16[128, 196, 512]" = torch.ops.aten.view.default(mm_109, _shape_param_456);  mm_109 = _shape_param_456 = None
        convert_element_type_340: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(view_466, torch.float32);  view_466 = None
        convert_element_type_341: "f32[2048, 512]" = torch.ops.prims.convert_element_type.default(mm_110, torch.float32);  mm_110 = None
        convert_element_type_342: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_339, torch.float32);  convert_element_type_339 = None
        mul_347: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_340, arg389_1);  arg389_1 = None
        mul_348: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_347, 512)
        sum_193: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_347, [2], True)
        convert_element_type_343: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(arg390_1, torch.float32);  arg390_1 = None
        sub_97: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_343, arg391_1);  convert_element_type_343 = arg391_1 = None
        mul_349: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_97, arg392_1);  sub_97 = None
        mul_350: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_347, mul_349);  mul_347 = None
        sum_194: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_350, [2], True);  mul_350 = None
        mul_351: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_349, sum_194);  sum_194 = None
        sub_98: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(mul_348, sum_193);  mul_348 = sum_193 = None
        sub_99: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(sub_98, mul_351);  sub_98 = mul_351 = None
        div_69: "f32[128, 196, 1]" = torch.ops.aten.div.Tensor(arg392_1, 512);  arg392_1 = None
        mul_352: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(div_69, sub_99);  div_69 = sub_99 = None
        mul_353: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_340, mul_349);  mul_349 = None
        sum_195: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_353, [0, 1]);  mul_353 = None
        sum_196: "f32[512]" = torch.ops.aten.sum.dim_IntList(convert_element_type_340, [0, 1]);  convert_element_type_340 = None
        convert_element_type_344: "bf16[128, 196, 512]" = torch.ops.prims.convert_element_type.default(mul_352, torch.bfloat16);  mul_352 = None
        add_74: "bf16[128, 196, 512]" = torch.ops.aten.add.Tensor(view_459, convert_element_type_344);  view_459 = convert_element_type_344 = None
        view_467: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(add_74, _shape_param_457);  add_74 = _shape_param_457 = None
        convert_element_type_345: "bf16[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(arg393_1, torch.bfloat16);  arg393_1 = None
        div_70: "bf16[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_345, 0.9565217345952988);  convert_element_type_345 = None
        mul_354: "bf16[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_467, div_70);  div_70 = None
        view_468: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(mul_354, _shape_param_458);  mul_354 = _shape_param_458 = None
        permute_147: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_468, [0, 1, 3, 2, 4, 5]);  view_468 = None
        clone_62: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_147, memory_format = torch.contiguous_format);  permute_147 = None
        view_469: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(clone_62, _shape_param_459);  clone_62 = _shape_param_459 = None
        view_470: "bf16[512, 49, 512]" = torch.ops.aten.view.default(view_469, _shape_param_460);  view_469 = _shape_param_460 = None
        view_471: "bf16[25088, 512]" = torch.ops.aten.view.default(view_470, _shape_param_461);  view_470 = _shape_param_461 = None
        mm_111: "bf16[25088, 512]" = torch.ops.aten.mm.default(view_471, arg394_1);  arg394_1 = None
        permute_148: "bf16[512, 25088]" = torch.ops.aten.permute.default(view_471, [1, 0])
        mm_112: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_148, arg395_1);  permute_148 = arg395_1 = None
        sum_197: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_471, [0], True, dtype = torch.float32);  view_471 = None
        view_472: "f32[512]" = torch.ops.aten.view.default(sum_197, _shape_param_462);  sum_197 = _shape_param_462 = None
        convert_element_type_346: "bf16[512]" = torch.ops.prims.convert_element_type.default(view_472, torch.bfloat16);  view_472 = None
        view_473: "bf16[512, 49, 512]" = torch.ops.aten.view.default(mm_111, _shape_param_463);  mm_111 = _shape_param_463 = None
        convert_element_type_347: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_112, torch.float32);  mm_112 = None
        convert_element_type_348: "f32[512]" = torch.ops.prims.convert_element_type.default(convert_element_type_346, torch.float32);  convert_element_type_346 = None
        view_474: "bf16[512, 49, 16, 32]" = torch.ops.aten.view.default(view_473, _shape_param_464);  view_473 = _shape_param_464 = None
        permute_149: "bf16[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_474, [0, 2, 1, 3]);  view_474 = None
        clone_63: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(permute_149, memory_format = torch.contiguous_format);  permute_149 = None
        view_475: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_63, _shape_param_465);  clone_63 = _shape_param_465 = None
        bmm_52: "bf16[8192, 49, 32]" = torch.ops.aten.bmm.default(arg396_1, view_475);  arg396_1 = None
        bmm_53: "bf16[8192, 49, 49]" = torch.ops.aten.bmm.default(view_475, arg397_1);  view_475 = arg397_1 = None
        view_476: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(bmm_52, _shape_param_466);  bmm_52 = _shape_param_466 = None
        view_477: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(bmm_53, _shape_param_467);  bmm_53 = _shape_param_467 = None
        convert_element_type_349: "f32[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(view_477, torch.float32);  view_477 = None
        slice_23: "bf16[8192, 49, 56]" = torch.ops.aten.slice.Tensor(arg398_1, 1, 0, -7);  arg398_1 = None
        slice_24: "bf16[8192, 49, 49]" = torch.ops.aten.slice.Tensor(slice_23, 2, 0, -7);  slice_23 = None
        view_478: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(slice_24, _shape_param_468);  slice_24 = _shape_param_468 = None
        view_479: "i64[2401]" = torch.ops.aten.view.default(arg399_1, [-1]);  arg399_1 = None
        index_37: "f32[2401, 16]" = torch.ops.aten.index.Tensor(arg400_1, [view_479]);  arg400_1 = None
        view_480: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_37, _shape_param_469);  index_37 = _shape_param_469 = None
        permute_150: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_480, [2, 0, 1]);  view_480 = None
        clone_64: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_150, memory_format = torch.contiguous_format);  permute_150 = None
        unsqueeze_27: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_64, 0);  clone_64 = None
        add_75: "f32[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_478, unsqueeze_27);  view_478 = unsqueeze_27 = None
        sub_100: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(add_75, arg401_1);  add_75 = arg401_1 = None
        exp_27: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_100);  sub_100 = None
        div_71: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_27, arg402_1);  exp_27 = arg402_1 = None
        mul_355: "f32[512, 16, 49, 49]" = torch.ops.aten.mul.Tensor(convert_element_type_349, div_71);  convert_element_type_349 = None
        sum_198: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_355, [-1], True)
        neg_13: "f32[512, 16, 49, 49]" = torch.ops.aten.neg.default(div_71);  div_71 = None
        fma_13: "f32[512, 16, 49, 49]" = torch.ops.prims.fma.default(neg_13, sum_198, mul_355);  neg_13 = sum_198 = mul_355 = None
        convert_element_type_350: "bf16[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(fma_13, torch.bfloat16)
        sum_199: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_13, [0], True, dtype = torch.float32);  fma_13 = None
        squeeze_13: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_199, 0);  sum_199 = None
        permute_151: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_13, [1, 2, 0]);  squeeze_13 = None
        view_481: "f32[2401, 16]" = torch.ops.aten.view.default(permute_151, _shape_param_470);  permute_151 = _shape_param_470 = None
        index_put_13: "f32[169, 16]" = torch.ops.aten.index_put.default(full_1, [view_479], view_481, True);  view_479 = view_481 = None
        view_482: "bf16[8192, 49, 49]" = torch.ops.aten.view.default(convert_element_type_350, _shape_param_471);  convert_element_type_350 = _shape_param_471 = None
        bmm_54: "bf16[8192, 32, 49]" = torch.ops.aten.bmm.default(arg403_1, view_482);  arg403_1 = None
        bmm_55: "bf16[8192, 49, 32]" = torch.ops.aten.bmm.default(view_482, arg404_1);  view_482 = arg404_1 = None
        view_483: "bf16[512, 16, 32, 49]" = torch.ops.aten.view.default(bmm_54, _shape_param_472);  bmm_54 = _shape_param_472 = None
        view_484: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(bmm_55, _shape_param_473);  bmm_55 = _shape_param_473 = None
        permute_152: "bf16[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_483, [0, 1, 3, 2]);  view_483 = None
        mul_356: "bf16[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_484, 0.1767766952966369);  view_484 = None
        cat_13: "bf16[1536, 16, 49, 32]" = torch.ops.aten.cat.default([mul_356, permute_152, view_476]);  mul_356 = permute_152 = view_476 = None
        view_485: "bf16[3, 512, 16, 49, 32]" = torch.ops.aten.view.default(cat_13, _shape_param_474);  cat_13 = _shape_param_474 = None
        permute_153: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.permute.default(view_485, [1, 3, 0, 2, 4]);  view_485 = None
        clone_65: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_153, memory_format = torch.contiguous_format);  permute_153 = None
        view_486: "bf16[512, 49, 1536]" = torch.ops.aten.view.default(clone_65, _shape_param_475);  clone_65 = _shape_param_475 = None
        view_487: "bf16[25088, 1536]" = torch.ops.aten.view.default(view_486, _shape_param_476);  view_486 = _shape_param_476 = None
        mm_113: "bf16[25088, 512]" = torch.ops.aten.mm.default(view_487, arg405_1);  arg405_1 = None
        permute_154: "bf16[1536, 25088]" = torch.ops.aten.permute.default(view_487, [1, 0])
        mm_114: "bf16[1536, 512]" = torch.ops.aten.mm.default(permute_154, arg406_1);  permute_154 = arg406_1 = None
        sum_200: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_487, [0], True, dtype = torch.float32);  view_487 = None
        view_488: "f32[1536]" = torch.ops.aten.view.default(sum_200, _shape_param_477);  sum_200 = _shape_param_477 = None
        convert_element_type_351: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_488, torch.bfloat16);  view_488 = None
        view_489: "bf16[512, 49, 512]" = torch.ops.aten.view.default(mm_113, _shape_param_478);  mm_113 = _shape_param_478 = None
        convert_element_type_352: "f32[512, 49, 512]" = torch.ops.prims.convert_element_type.default(view_489, torch.float32);  view_489 = None
        convert_element_type_353: "f32[1536, 512]" = torch.ops.prims.convert_element_type.default(mm_114, torch.float32);  mm_114 = None
        convert_element_type_354: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_351, torch.float32);  convert_element_type_351 = None
        view_490: "f32[512, 7, 7, 512]" = torch.ops.aten.view.default(convert_element_type_352, _shape_param_479);  convert_element_type_352 = _shape_param_479 = None
        view_491: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_490, _shape_param_480);  view_490 = _shape_param_480 = None
        permute_155: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_491, [0, 1, 3, 2, 4, 5]);  view_491 = None
        clone_66: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_155, memory_format = torch.contiguous_format);  permute_155 = None
        view_492: "f32[128, 14, 14, 512]" = torch.ops.aten.view.default(clone_66, _shape_param_481);  clone_66 = _shape_param_481 = None
        mul_357: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_492, arg407_1);  arg407_1 = None
        mul_358: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_357, 512)
        sum_201: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_357, [3], True)
        convert_element_type_355: "f32[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(arg408_1, torch.float32);  arg408_1 = None
        sub_101: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_355, arg409_1);  convert_element_type_355 = arg409_1 = None
        mul_359: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_101, arg410_1);  sub_101 = None
        mul_360: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_357, mul_359);  mul_357 = None
        sum_202: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_360, [3], True);  mul_360 = None
        mul_361: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_359, sum_202);  sum_202 = None
        sub_102: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_358, sum_201);  mul_358 = sum_201 = None
        sub_103: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_102, mul_361);  sub_102 = mul_361 = None
        div_72: "f32[128, 14, 14, 1]" = torch.ops.aten.div.Tensor(arg410_1, 512);  arg410_1 = None
        mul_362: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_72, sub_103);  div_72 = sub_103 = None
        mul_363: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_492, mul_359);  mul_359 = None
        sum_203: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_363, [0, 1, 2]);  mul_363 = None
        sum_204: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_492, [0, 1, 2]);  view_492 = None
        convert_element_type_356: "bf16[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(mul_362, torch.bfloat16);  mul_362 = None
        add_76: "bf16[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_467, convert_element_type_356);  view_467 = convert_element_type_356 = None
        view_493: "bf16[128, 196, 512]" = torch.ops.aten.view.default(add_76, _shape_param_482);  add_76 = _shape_param_482 = None
        convert_element_type_357: "bf16[128, 1, 1]" = torch.ops.prims.convert_element_type.default(arg411_1, torch.bfloat16);  arg411_1 = None
        div_73: "bf16[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_357, 0.960869561880827);  convert_element_type_357 = None
        mul_364: "bf16[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_493, div_73);  div_73 = None
        view_494: "bf16[25088, 512]" = torch.ops.aten.view.default(mul_364, _shape_param_483);  mul_364 = _shape_param_483 = None
        mm_115: "bf16[25088, 2048]" = torch.ops.aten.mm.default(view_494, arg412_1);  arg412_1 = None
        permute_156: "bf16[512, 25088]" = torch.ops.aten.permute.default(view_494, [1, 0])
        mm_116: "bf16[512, 2048]" = torch.ops.aten.mm.default(permute_156, arg413_1);  permute_156 = arg413_1 = None
        sum_205: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_494, [0], True, dtype = torch.float32);  view_494 = None
        view_495: "f32[512]" = torch.ops.aten.view.default(sum_205, _shape_param_484);  sum_205 = _shape_param_484 = None
        convert_element_type_358: "bf16[512]" = torch.ops.prims.convert_element_type.default(view_495, torch.bfloat16);  view_495 = None
        view_496: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(mm_115, _shape_param_485);  mm_115 = _shape_param_485 = None
        convert_element_type_359: "f32[512, 2048]" = torch.ops.prims.convert_element_type.default(mm_116, torch.float32);  mm_116 = None
        convert_element_type_360: "f32[512]" = torch.ops.prims.convert_element_type.default(convert_element_type_358, torch.float32);  convert_element_type_358 = None
        convert_element_type_361: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_496, torch.float32);  view_496 = None
        view_497: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(arg414_1, _shape_param_486);  arg414_1 = _shape_param_486 = None
        convert_element_type_362: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_497, torch.float32);  view_497 = None
        mul_365: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_362, 0.7071067811865476)
        erf_14: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_365);  mul_365 = None
        add_77: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
        mul_366: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(add_77, 0.5);  add_77 = None
        mul_367: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_362, convert_element_type_362)
        mul_368: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_367, -0.5);  mul_367 = None
        exp_28: "f32[128, 196, 2048]" = torch.ops.aten.exp.default(mul_368);  mul_368 = None
        mul_369: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_28, 0.3989422804014327);  exp_28 = None
        mul_370: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_362, mul_369);  convert_element_type_362 = mul_369 = None
        add_78: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(mul_366, mul_370);  mul_366 = mul_370 = None
        mul_371: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_361, add_78);  convert_element_type_361 = add_78 = None
        convert_element_type_363: "bf16[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(mul_371, torch.bfloat16);  mul_371 = None
        view_498: "bf16[25088, 2048]" = torch.ops.aten.view.default(convert_element_type_363, _shape_param_487);  convert_element_type_363 = _shape_param_487 = None
        mm_117: "bf16[25088, 512]" = torch.ops.aten.mm.default(view_498, arg415_1);  arg415_1 = None
        permute_157: "bf16[2048, 25088]" = torch.ops.aten.permute.default(view_498, [1, 0])
        mm_118: "bf16[2048, 512]" = torch.ops.aten.mm.default(permute_157, arg416_1);  permute_157 = arg416_1 = None
        sum_206: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_498, [0], True, dtype = torch.float32);  view_498 = None
        view_499: "f32[2048]" = torch.ops.aten.view.default(sum_206, _shape_param_488);  sum_206 = _shape_param_488 = None
        convert_element_type_364: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_499, torch.bfloat16);  view_499 = None
        view_500: "bf16[128, 196, 512]" = torch.ops.aten.view.default(mm_117, _shape_param_489);  mm_117 = _shape_param_489 = None
        convert_element_type_365: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(view_500, torch.float32);  view_500 = None
        convert_element_type_366: "f32[2048, 512]" = torch.ops.prims.convert_element_type.default(mm_118, torch.float32);  mm_118 = None
        convert_element_type_367: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_364, torch.float32);  convert_element_type_364 = None
        mul_372: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_365, arg417_1);  arg417_1 = None
        mul_373: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_372, 512)
        sum_207: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_372, [2], True)
        convert_element_type_368: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(arg418_1, torch.float32);  arg418_1 = None
        sub_104: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_368, arg419_1);  convert_element_type_368 = arg419_1 = None
        mul_374: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_104, arg420_1);  sub_104 = None
        mul_375: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_372, mul_374);  mul_372 = None
        sum_208: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_375, [2], True);  mul_375 = None
        mul_376: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_374, sum_208);  sum_208 = None
        sub_105: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(mul_373, sum_207);  mul_373 = sum_207 = None
        sub_106: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(sub_105, mul_376);  sub_105 = mul_376 = None
        div_74: "f32[128, 196, 1]" = torch.ops.aten.div.Tensor(arg420_1, 512);  arg420_1 = None
        mul_377: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(div_74, sub_106);  div_74 = sub_106 = None
        mul_378: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_365, mul_374);  mul_374 = None
        sum_209: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_378, [0, 1]);  mul_378 = None
        sum_210: "f32[512]" = torch.ops.aten.sum.dim_IntList(convert_element_type_365, [0, 1]);  convert_element_type_365 = None
        convert_element_type_369: "bf16[128, 196, 512]" = torch.ops.prims.convert_element_type.default(mul_377, torch.bfloat16);  mul_377 = None
        add_79: "bf16[128, 196, 512]" = torch.ops.aten.add.Tensor(view_493, convert_element_type_369);  view_493 = convert_element_type_369 = None
        view_501: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(add_79, _shape_param_490);  add_79 = _shape_param_490 = None
        convert_element_type_370: "bf16[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(arg421_1, torch.bfloat16);  arg421_1 = None
        div_75: "bf16[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_370, 0.960869561880827);  convert_element_type_370 = None
        mul_379: "bf16[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_501, div_75);  div_75 = None
        index_38: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(mul_379, [None, None, fmod]);  mul_379 = None
        index_39: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_38, [None, fmod]);  index_38 = None
        view_502: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(index_39, _shape_param_491);  index_39 = _shape_param_491 = None
        permute_158: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_502, [0, 1, 3, 2, 4, 5]);  view_502 = None
        clone_67: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_158, memory_format = torch.contiguous_format);  permute_158 = None
        view_503: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(clone_67, _shape_param_492);  clone_67 = _shape_param_492 = None
        view_504: "bf16[512, 49, 512]" = torch.ops.aten.view.default(view_503, _shape_param_493);  view_503 = _shape_param_493 = None
        view_505: "bf16[25088, 512]" = torch.ops.aten.view.default(view_504, _shape_param_494);  view_504 = _shape_param_494 = None
        mm_119: "bf16[25088, 512]" = torch.ops.aten.mm.default(view_505, arg422_1);  arg422_1 = None
        permute_159: "bf16[512, 25088]" = torch.ops.aten.permute.default(view_505, [1, 0])
        mm_120: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_159, arg423_1);  permute_159 = arg423_1 = None
        sum_211: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_505, [0], True, dtype = torch.float32);  view_505 = None
        view_506: "f32[512]" = torch.ops.aten.view.default(sum_211, _shape_param_495);  sum_211 = _shape_param_495 = None
        convert_element_type_371: "bf16[512]" = torch.ops.prims.convert_element_type.default(view_506, torch.bfloat16);  view_506 = None
        view_507: "bf16[512, 49, 512]" = torch.ops.aten.view.default(mm_119, _shape_param_496);  mm_119 = _shape_param_496 = None
        convert_element_type_372: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_120, torch.float32);  mm_120 = None
        convert_element_type_373: "f32[512]" = torch.ops.prims.convert_element_type.default(convert_element_type_371, torch.float32);  convert_element_type_371 = None
        view_508: "bf16[512, 49, 16, 32]" = torch.ops.aten.view.default(view_507, _shape_param_497);  view_507 = _shape_param_497 = None
        permute_160: "bf16[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_508, [0, 2, 1, 3]);  view_508 = None
        clone_68: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(permute_160, memory_format = torch.contiguous_format);  permute_160 = None
        view_509: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_68, _shape_param_498);  clone_68 = _shape_param_498 = None
        bmm_56: "bf16[8192, 49, 32]" = torch.ops.aten.bmm.default(arg424_1, view_509);  arg424_1 = None
        bmm_57: "bf16[8192, 49, 49]" = torch.ops.aten.bmm.default(view_509, arg425_1);  view_509 = arg425_1 = None
        view_510: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(bmm_56, _shape_param_499);  bmm_56 = _shape_param_499 = None
        view_511: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(bmm_57, _shape_param_500);  bmm_57 = _shape_param_500 = None
        convert_element_type_374: "f32[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(view_511, torch.float32);  view_511 = None
        slice_25: "bf16[8192, 49, 56]" = torch.ops.aten.slice.Tensor(arg426_1, 1, 0, -7);  arg426_1 = None
        slice_26: "bf16[8192, 49, 49]" = torch.ops.aten.slice.Tensor(slice_25, 2, 0, -7);  slice_25 = None
        view_512: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(slice_26, _shape_param_501);  slice_26 = _shape_param_501 = None
        view_513: "i64[2401]" = torch.ops.aten.view.default(arg427_1, [-1]);  arg427_1 = None
        index_40: "f32[2401, 16]" = torch.ops.aten.index.Tensor(arg428_1, [view_513]);  arg428_1 = None
        view_514: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_40, _shape_param_502);  index_40 = _shape_param_502 = None
        permute_161: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_514, [2, 0, 1]);  view_514 = None
        clone_69: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_161, memory_format = torch.contiguous_format);  permute_161 = None
        unsqueeze_28: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_69, 0);  clone_69 = None
        add_80: "f32[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_512, unsqueeze_28);  view_512 = unsqueeze_28 = None
        view_515: "f32[128, 4, 16, 49, 49]" = torch.ops.aten.view.default(add_80, _shape_param_503);  add_80 = _shape_param_503 = None
        unsqueeze_29: "f32[4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(arg429_1, 1);  arg429_1 = None
        unsqueeze_30: "f32[1, 4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_29, 0);  unsqueeze_29 = None
        add_81: "f32[128, 4, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_515, unsqueeze_30);  view_515 = unsqueeze_30 = None
        view_516: "f32[512, 16, 49, 49]" = torch.ops.aten.view.default(add_81, _shape_param_504);  add_81 = _shape_param_504 = None
        sub_107: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(view_516, arg430_1);  view_516 = arg430_1 = None
        exp_29: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_107);  sub_107 = None
        div_76: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_29, arg431_1);  exp_29 = arg431_1 = None
        mul_380: "f32[512, 16, 49, 49]" = torch.ops.aten.mul.Tensor(convert_element_type_374, div_76);  convert_element_type_374 = None
        sum_212: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_380, [-1], True)
        neg_14: "f32[512, 16, 49, 49]" = torch.ops.aten.neg.default(div_76);  div_76 = None
        fma_14: "f32[512, 16, 49, 49]" = torch.ops.prims.fma.default(neg_14, sum_212, mul_380);  neg_14 = sum_212 = mul_380 = None
        convert_element_type_375: "bf16[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(fma_14, torch.bfloat16)
        sum_213: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_14, [0], True, dtype = torch.float32);  fma_14 = None
        squeeze_14: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_213, 0);  sum_213 = None
        permute_162: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_14, [1, 2, 0]);  squeeze_14 = None
        view_517: "f32[2401, 16]" = torch.ops.aten.view.default(permute_162, _shape_param_505);  permute_162 = _shape_param_505 = None
        index_put_14: "f32[169, 16]" = torch.ops.aten.index_put.default(full_1, [view_513], view_517, True);  view_513 = view_517 = None
        view_518: "bf16[8192, 49, 49]" = torch.ops.aten.view.default(convert_element_type_375, _shape_param_506);  convert_element_type_375 = _shape_param_506 = None
        bmm_58: "bf16[8192, 32, 49]" = torch.ops.aten.bmm.default(arg432_1, view_518);  arg432_1 = None
        bmm_59: "bf16[8192, 49, 32]" = torch.ops.aten.bmm.default(view_518, arg433_1);  view_518 = arg433_1 = None
        view_519: "bf16[512, 16, 32, 49]" = torch.ops.aten.view.default(bmm_58, _shape_param_507);  bmm_58 = _shape_param_507 = None
        view_520: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(bmm_59, _shape_param_508);  bmm_59 = _shape_param_508 = None
        permute_163: "bf16[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_519, [0, 1, 3, 2]);  view_519 = None
        mul_381: "bf16[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_520, 0.1767766952966369);  view_520 = None
        cat_14: "bf16[1536, 16, 49, 32]" = torch.ops.aten.cat.default([mul_381, permute_163, view_510]);  mul_381 = permute_163 = view_510 = None
        view_521: "bf16[3, 512, 16, 49, 32]" = torch.ops.aten.view.default(cat_14, _shape_param_509);  cat_14 = _shape_param_509 = None
        permute_164: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.permute.default(view_521, [1, 3, 0, 2, 4]);  view_521 = None
        clone_70: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_164, memory_format = torch.contiguous_format);  permute_164 = None
        view_522: "bf16[512, 49, 1536]" = torch.ops.aten.view.default(clone_70, _shape_param_510);  clone_70 = _shape_param_510 = None
        view_523: "bf16[25088, 1536]" = torch.ops.aten.view.default(view_522, _shape_param_511);  view_522 = _shape_param_511 = None
        mm_121: "bf16[25088, 512]" = torch.ops.aten.mm.default(view_523, arg434_1);  arg434_1 = None
        permute_165: "bf16[1536, 25088]" = torch.ops.aten.permute.default(view_523, [1, 0])
        mm_122: "bf16[1536, 512]" = torch.ops.aten.mm.default(permute_165, arg435_1);  permute_165 = arg435_1 = None
        sum_214: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_523, [0], True, dtype = torch.float32);  view_523 = None
        view_524: "f32[1536]" = torch.ops.aten.view.default(sum_214, _shape_param_512);  sum_214 = _shape_param_512 = None
        convert_element_type_376: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_524, torch.bfloat16);  view_524 = None
        view_525: "bf16[512, 49, 512]" = torch.ops.aten.view.default(mm_121, _shape_param_513);  mm_121 = _shape_param_513 = None
        convert_element_type_377: "f32[512, 49, 512]" = torch.ops.prims.convert_element_type.default(view_525, torch.float32);  view_525 = None
        convert_element_type_378: "f32[1536, 512]" = torch.ops.prims.convert_element_type.default(mm_122, torch.float32);  mm_122 = None
        convert_element_type_379: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_376, torch.float32);  convert_element_type_376 = None
        view_526: "f32[512, 7, 7, 512]" = torch.ops.aten.view.default(convert_element_type_377, _shape_param_514);  convert_element_type_377 = _shape_param_514 = None
        view_527: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_526, _shape_param_515);  view_526 = _shape_param_515 = None
        permute_166: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_527, [0, 1, 3, 2, 4, 5]);  view_527 = None
        clone_71: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_166, memory_format = torch.contiguous_format);  permute_166 = None
        view_528: "f32[128, 14, 14, 512]" = torch.ops.aten.view.default(clone_71, _shape_param_516);  clone_71 = _shape_param_516 = None
        index_41: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(view_528, [None, None, arg93_1]);  view_528 = None
        index_42: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_41, [None, arg93_1]);  index_41 = None
        mul_382: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(index_42, arg436_1);  arg436_1 = None
        mul_383: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_382, 512)
        sum_215: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_382, [3], True)
        convert_element_type_380: "f32[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(arg437_1, torch.float32);  arg437_1 = None
        sub_108: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_380, arg438_1);  convert_element_type_380 = arg438_1 = None
        mul_384: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_108, arg439_1);  sub_108 = None
        mul_385: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_382, mul_384);  mul_382 = None
        sum_216: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_385, [3], True);  mul_385 = None
        mul_386: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_384, sum_216);  sum_216 = None
        sub_109: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_383, sum_215);  mul_383 = sum_215 = None
        sub_110: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_109, mul_386);  sub_109 = mul_386 = None
        div_77: "f32[128, 14, 14, 1]" = torch.ops.aten.div.Tensor(arg439_1, 512);  arg439_1 = None
        mul_387: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_77, sub_110);  div_77 = sub_110 = None
        mul_388: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(index_42, mul_384);  mul_384 = None
        sum_217: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_388, [0, 1, 2]);  mul_388 = None
        sum_218: "f32[512]" = torch.ops.aten.sum.dim_IntList(index_42, [0, 1, 2]);  index_42 = None
        convert_element_type_381: "bf16[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(mul_387, torch.bfloat16);  mul_387 = None
        add_82: "bf16[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_501, convert_element_type_381);  view_501 = convert_element_type_381 = None
        view_529: "bf16[128, 196, 512]" = torch.ops.aten.view.default(add_82, _shape_param_517);  add_82 = _shape_param_517 = None
        convert_element_type_382: "bf16[128, 1, 1]" = torch.ops.prims.convert_element_type.default(arg440_1, torch.bfloat16);  arg440_1 = None
        div_78: "bf16[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_382, 0.9652173891663551);  convert_element_type_382 = None
        mul_389: "bf16[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_529, div_78);  div_78 = None
        view_530: "bf16[25088, 512]" = torch.ops.aten.view.default(mul_389, _shape_param_518);  mul_389 = _shape_param_518 = None
        mm_123: "bf16[25088, 2048]" = torch.ops.aten.mm.default(view_530, arg441_1);  arg441_1 = None
        permute_167: "bf16[512, 25088]" = torch.ops.aten.permute.default(view_530, [1, 0])
        mm_124: "bf16[512, 2048]" = torch.ops.aten.mm.default(permute_167, arg442_1);  permute_167 = arg442_1 = None
        sum_219: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_530, [0], True, dtype = torch.float32);  view_530 = None
        view_531: "f32[512]" = torch.ops.aten.view.default(sum_219, _shape_param_519);  sum_219 = _shape_param_519 = None
        convert_element_type_383: "bf16[512]" = torch.ops.prims.convert_element_type.default(view_531, torch.bfloat16);  view_531 = None
        view_532: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(mm_123, _shape_param_520);  mm_123 = _shape_param_520 = None
        convert_element_type_384: "f32[512, 2048]" = torch.ops.prims.convert_element_type.default(mm_124, torch.float32);  mm_124 = None
        convert_element_type_385: "f32[512]" = torch.ops.prims.convert_element_type.default(convert_element_type_383, torch.float32);  convert_element_type_383 = None
        convert_element_type_386: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_532, torch.float32);  view_532 = None
        view_533: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(arg443_1, _shape_param_521);  arg443_1 = _shape_param_521 = None
        convert_element_type_387: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_533, torch.float32);  view_533 = None
        mul_390: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_387, 0.7071067811865476)
        erf_15: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_390);  mul_390 = None
        add_83: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
        mul_391: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(add_83, 0.5);  add_83 = None
        mul_392: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_387, convert_element_type_387)
        mul_393: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_392, -0.5);  mul_392 = None
        exp_30: "f32[128, 196, 2048]" = torch.ops.aten.exp.default(mul_393);  mul_393 = None
        mul_394: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_30, 0.3989422804014327);  exp_30 = None
        mul_395: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_387, mul_394);  convert_element_type_387 = mul_394 = None
        add_84: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(mul_391, mul_395);  mul_391 = mul_395 = None
        mul_396: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_386, add_84);  convert_element_type_386 = add_84 = None
        convert_element_type_388: "bf16[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(mul_396, torch.bfloat16);  mul_396 = None
        view_534: "bf16[25088, 2048]" = torch.ops.aten.view.default(convert_element_type_388, _shape_param_522);  convert_element_type_388 = _shape_param_522 = None
        mm_125: "bf16[25088, 512]" = torch.ops.aten.mm.default(view_534, arg444_1);  arg444_1 = None
        permute_168: "bf16[2048, 25088]" = torch.ops.aten.permute.default(view_534, [1, 0])
        mm_126: "bf16[2048, 512]" = torch.ops.aten.mm.default(permute_168, arg445_1);  permute_168 = arg445_1 = None
        sum_220: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_534, [0], True, dtype = torch.float32);  view_534 = None
        view_535: "f32[2048]" = torch.ops.aten.view.default(sum_220, _shape_param_523);  sum_220 = _shape_param_523 = None
        convert_element_type_389: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_535, torch.bfloat16);  view_535 = None
        view_536: "bf16[128, 196, 512]" = torch.ops.aten.view.default(mm_125, _shape_param_524);  mm_125 = _shape_param_524 = None
        convert_element_type_390: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(view_536, torch.float32);  view_536 = None
        convert_element_type_391: "f32[2048, 512]" = torch.ops.prims.convert_element_type.default(mm_126, torch.float32);  mm_126 = None
        convert_element_type_392: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_389, torch.float32);  convert_element_type_389 = None
        mul_397: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_390, arg446_1);  arg446_1 = None
        mul_398: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_397, 512)
        sum_221: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_397, [2], True)
        convert_element_type_393: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(arg447_1, torch.float32);  arg447_1 = None
        sub_111: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_393, arg448_1);  convert_element_type_393 = arg448_1 = None
        mul_399: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_111, arg449_1);  sub_111 = None
        mul_400: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_397, mul_399);  mul_397 = None
        sum_222: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_400, [2], True);  mul_400 = None
        mul_401: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_399, sum_222);  sum_222 = None
        sub_112: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(mul_398, sum_221);  mul_398 = sum_221 = None
        sub_113: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(sub_112, mul_401);  sub_112 = mul_401 = None
        div_79: "f32[128, 196, 1]" = torch.ops.aten.div.Tensor(arg449_1, 512);  arg449_1 = None
        mul_402: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(div_79, sub_113);  div_79 = sub_113 = None
        mul_403: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_390, mul_399);  mul_399 = None
        sum_223: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_403, [0, 1]);  mul_403 = None
        sum_224: "f32[512]" = torch.ops.aten.sum.dim_IntList(convert_element_type_390, [0, 1]);  convert_element_type_390 = None
        convert_element_type_394: "bf16[128, 196, 512]" = torch.ops.prims.convert_element_type.default(mul_402, torch.bfloat16);  mul_402 = None
        add_85: "bf16[128, 196, 512]" = torch.ops.aten.add.Tensor(view_529, convert_element_type_394);  view_529 = convert_element_type_394 = None
        view_537: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(add_85, _shape_param_525);  add_85 = _shape_param_525 = None
        convert_element_type_395: "bf16[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(arg450_1, torch.bfloat16);  arg450_1 = None
        div_80: "bf16[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_395, 0.9652173891663551);  convert_element_type_395 = None
        mul_404: "bf16[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_537, div_80);  div_80 = None
        view_538: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(mul_404, _shape_param_526);  mul_404 = _shape_param_526 = None
        permute_169: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_538, [0, 1, 3, 2, 4, 5]);  view_538 = None
        clone_72: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_169, memory_format = torch.contiguous_format);  permute_169 = None
        view_539: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(clone_72, _shape_param_527);  clone_72 = _shape_param_527 = None
        view_540: "bf16[512, 49, 512]" = torch.ops.aten.view.default(view_539, _shape_param_528);  view_539 = _shape_param_528 = None
        view_541: "bf16[25088, 512]" = torch.ops.aten.view.default(view_540, _shape_param_529);  view_540 = _shape_param_529 = None
        mm_127: "bf16[25088, 512]" = torch.ops.aten.mm.default(view_541, arg451_1);  arg451_1 = None
        permute_170: "bf16[512, 25088]" = torch.ops.aten.permute.default(view_541, [1, 0])
        mm_128: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_170, arg452_1);  permute_170 = arg452_1 = None
        sum_225: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_541, [0], True, dtype = torch.float32);  view_541 = None
        view_542: "f32[512]" = torch.ops.aten.view.default(sum_225, _shape_param_530);  sum_225 = _shape_param_530 = None
        convert_element_type_396: "bf16[512]" = torch.ops.prims.convert_element_type.default(view_542, torch.bfloat16);  view_542 = None
        view_543: "bf16[512, 49, 512]" = torch.ops.aten.view.default(mm_127, _shape_param_531);  mm_127 = _shape_param_531 = None
        convert_element_type_397: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_128, torch.float32);  mm_128 = None
        convert_element_type_398: "f32[512]" = torch.ops.prims.convert_element_type.default(convert_element_type_396, torch.float32);  convert_element_type_396 = None
        view_544: "bf16[512, 49, 16, 32]" = torch.ops.aten.view.default(view_543, _shape_param_532);  view_543 = _shape_param_532 = None
        permute_171: "bf16[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_544, [0, 2, 1, 3]);  view_544 = None
        clone_73: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(permute_171, memory_format = torch.contiguous_format);  permute_171 = None
        view_545: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_73, _shape_param_533);  clone_73 = _shape_param_533 = None
        bmm_60: "bf16[8192, 49, 32]" = torch.ops.aten.bmm.default(arg453_1, view_545);  arg453_1 = None
        bmm_61: "bf16[8192, 49, 49]" = torch.ops.aten.bmm.default(view_545, arg454_1);  view_545 = arg454_1 = None
        view_546: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(bmm_60, _shape_param_534);  bmm_60 = _shape_param_534 = None
        view_547: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(bmm_61, _shape_param_535);  bmm_61 = _shape_param_535 = None
        convert_element_type_399: "f32[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(view_547, torch.float32);  view_547 = None
        slice_27: "bf16[8192, 49, 56]" = torch.ops.aten.slice.Tensor(arg455_1, 1, 0, -7);  arg455_1 = None
        slice_28: "bf16[8192, 49, 49]" = torch.ops.aten.slice.Tensor(slice_27, 2, 0, -7);  slice_27 = None
        view_548: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(slice_28, _shape_param_536);  slice_28 = _shape_param_536 = None
        view_549: "i64[2401]" = torch.ops.aten.view.default(arg456_1, [-1]);  arg456_1 = None
        index_43: "f32[2401, 16]" = torch.ops.aten.index.Tensor(arg457_1, [view_549]);  arg457_1 = None
        view_550: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_43, _shape_param_537);  index_43 = _shape_param_537 = None
        permute_172: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_550, [2, 0, 1]);  view_550 = None
        clone_74: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_172, memory_format = torch.contiguous_format);  permute_172 = None
        unsqueeze_31: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_74, 0);  clone_74 = None
        add_86: "f32[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_548, unsqueeze_31);  view_548 = unsqueeze_31 = None
        sub_114: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(add_86, arg458_1);  add_86 = arg458_1 = None
        exp_31: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_114);  sub_114 = None
        div_81: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_31, arg459_1);  exp_31 = arg459_1 = None
        mul_405: "f32[512, 16, 49, 49]" = torch.ops.aten.mul.Tensor(convert_element_type_399, div_81);  convert_element_type_399 = None
        sum_226: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_405, [-1], True)
        neg_15: "f32[512, 16, 49, 49]" = torch.ops.aten.neg.default(div_81);  div_81 = None
        fma_15: "f32[512, 16, 49, 49]" = torch.ops.prims.fma.default(neg_15, sum_226, mul_405);  neg_15 = sum_226 = mul_405 = None
        convert_element_type_400: "bf16[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(fma_15, torch.bfloat16)
        sum_227: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_15, [0], True, dtype = torch.float32);  fma_15 = None
        squeeze_15: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_227, 0);  sum_227 = None
        permute_173: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_15, [1, 2, 0]);  squeeze_15 = None
        view_551: "f32[2401, 16]" = torch.ops.aten.view.default(permute_173, _shape_param_538);  permute_173 = _shape_param_538 = None
        index_put_15: "f32[169, 16]" = torch.ops.aten.index_put.default(full_1, [view_549], view_551, True);  view_549 = view_551 = None
        view_552: "bf16[8192, 49, 49]" = torch.ops.aten.view.default(convert_element_type_400, _shape_param_539);  convert_element_type_400 = _shape_param_539 = None
        bmm_62: "bf16[8192, 32, 49]" = torch.ops.aten.bmm.default(arg460_1, view_552);  arg460_1 = None
        bmm_63: "bf16[8192, 49, 32]" = torch.ops.aten.bmm.default(view_552, arg461_1);  view_552 = arg461_1 = None
        view_553: "bf16[512, 16, 32, 49]" = torch.ops.aten.view.default(bmm_62, _shape_param_540);  bmm_62 = _shape_param_540 = None
        view_554: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(bmm_63, _shape_param_541);  bmm_63 = _shape_param_541 = None
        permute_174: "bf16[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_553, [0, 1, 3, 2]);  view_553 = None
        mul_406: "bf16[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_554, 0.1767766952966369);  view_554 = None
        cat_15: "bf16[1536, 16, 49, 32]" = torch.ops.aten.cat.default([mul_406, permute_174, view_546]);  mul_406 = permute_174 = view_546 = None
        view_555: "bf16[3, 512, 16, 49, 32]" = torch.ops.aten.view.default(cat_15, _shape_param_542);  cat_15 = _shape_param_542 = None
        permute_175: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.permute.default(view_555, [1, 3, 0, 2, 4]);  view_555 = None
        clone_75: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_175, memory_format = torch.contiguous_format);  permute_175 = None
        view_556: "bf16[512, 49, 1536]" = torch.ops.aten.view.default(clone_75, _shape_param_543);  clone_75 = _shape_param_543 = None
        view_557: "bf16[25088, 1536]" = torch.ops.aten.view.default(view_556, _shape_param_544);  view_556 = _shape_param_544 = None
        mm_129: "bf16[25088, 512]" = torch.ops.aten.mm.default(view_557, arg462_1);  arg462_1 = None
        permute_176: "bf16[1536, 25088]" = torch.ops.aten.permute.default(view_557, [1, 0])
        mm_130: "bf16[1536, 512]" = torch.ops.aten.mm.default(permute_176, arg463_1);  permute_176 = arg463_1 = None
        sum_228: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_557, [0], True, dtype = torch.float32);  view_557 = None
        view_558: "f32[1536]" = torch.ops.aten.view.default(sum_228, _shape_param_545);  sum_228 = _shape_param_545 = None
        convert_element_type_401: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_558, torch.bfloat16);  view_558 = None
        view_559: "bf16[512, 49, 512]" = torch.ops.aten.view.default(mm_129, _shape_param_546);  mm_129 = _shape_param_546 = None
        convert_element_type_402: "f32[512, 49, 512]" = torch.ops.prims.convert_element_type.default(view_559, torch.float32);  view_559 = None
        convert_element_type_403: "f32[1536, 512]" = torch.ops.prims.convert_element_type.default(mm_130, torch.float32);  mm_130 = None
        convert_element_type_404: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_401, torch.float32);  convert_element_type_401 = None
        view_560: "f32[512, 7, 7, 512]" = torch.ops.aten.view.default(convert_element_type_402, _shape_param_547);  convert_element_type_402 = _shape_param_547 = None
        view_561: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_560, _shape_param_548);  view_560 = _shape_param_548 = None
        permute_177: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_561, [0, 1, 3, 2, 4, 5]);  view_561 = None
        clone_76: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_177, memory_format = torch.contiguous_format);  permute_177 = None
        view_562: "f32[128, 14, 14, 512]" = torch.ops.aten.view.default(clone_76, _shape_param_549);  clone_76 = _shape_param_549 = None
        mul_407: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_562, arg464_1);  arg464_1 = None
        mul_408: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_407, 512)
        sum_229: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_407, [3], True)
        convert_element_type_405: "f32[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(arg465_1, torch.float32);  arg465_1 = None
        sub_115: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_405, arg466_1);  convert_element_type_405 = arg466_1 = None
        mul_409: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_115, arg467_1);  sub_115 = None
        mul_410: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_407, mul_409);  mul_407 = None
        sum_230: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_410, [3], True);  mul_410 = None
        mul_411: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_409, sum_230);  sum_230 = None
        sub_116: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_408, sum_229);  mul_408 = sum_229 = None
        sub_117: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_116, mul_411);  sub_116 = mul_411 = None
        div_82: "f32[128, 14, 14, 1]" = torch.ops.aten.div.Tensor(arg467_1, 512);  arg467_1 = None
        mul_412: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_82, sub_117);  div_82 = sub_117 = None
        mul_413: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_562, mul_409);  mul_409 = None
        sum_231: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_413, [0, 1, 2]);  mul_413 = None
        sum_232: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_562, [0, 1, 2]);  view_562 = None
        convert_element_type_406: "bf16[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(mul_412, torch.bfloat16);  mul_412 = None
        add_87: "bf16[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_537, convert_element_type_406);  view_537 = convert_element_type_406 = None
        view_563: "bf16[128, 196, 512]" = torch.ops.aten.view.default(add_87, _shape_param_550);  add_87 = _shape_param_550 = None
        convert_element_type_407: "bf16[128, 1, 1]" = torch.ops.prims.convert_element_type.default(arg468_1, torch.bfloat16);  arg468_1 = None
        div_83: "bf16[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_407, 0.9695652164518833);  convert_element_type_407 = None
        mul_414: "bf16[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_563, div_83);  div_83 = None
        view_564: "bf16[25088, 512]" = torch.ops.aten.view.default(mul_414, _shape_param_551);  mul_414 = _shape_param_551 = None
        mm_131: "bf16[25088, 2048]" = torch.ops.aten.mm.default(view_564, arg469_1);  arg469_1 = None
        permute_178: "bf16[512, 25088]" = torch.ops.aten.permute.default(view_564, [1, 0])
        mm_132: "bf16[512, 2048]" = torch.ops.aten.mm.default(permute_178, arg470_1);  permute_178 = arg470_1 = None
        sum_233: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_564, [0], True, dtype = torch.float32);  view_564 = None
        view_565: "f32[512]" = torch.ops.aten.view.default(sum_233, _shape_param_552);  sum_233 = _shape_param_552 = None
        convert_element_type_408: "bf16[512]" = torch.ops.prims.convert_element_type.default(view_565, torch.bfloat16);  view_565 = None
        view_566: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(mm_131, _shape_param_553);  mm_131 = _shape_param_553 = None
        convert_element_type_409: "f32[512, 2048]" = torch.ops.prims.convert_element_type.default(mm_132, torch.float32);  mm_132 = None
        convert_element_type_410: "f32[512]" = torch.ops.prims.convert_element_type.default(convert_element_type_408, torch.float32);  convert_element_type_408 = None
        convert_element_type_411: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_566, torch.float32);  view_566 = None
        view_567: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(arg471_1, _shape_param_554);  arg471_1 = _shape_param_554 = None
        convert_element_type_412: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_567, torch.float32);  view_567 = None
        mul_415: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_412, 0.7071067811865476)
        erf_16: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_415);  mul_415 = None
        add_88: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
        mul_416: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(add_88, 0.5);  add_88 = None
        mul_417: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_412, convert_element_type_412)
        mul_418: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_417, -0.5);  mul_417 = None
        exp_32: "f32[128, 196, 2048]" = torch.ops.aten.exp.default(mul_418);  mul_418 = None
        mul_419: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_32, 0.3989422804014327);  exp_32 = None
        mul_420: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_412, mul_419);  convert_element_type_412 = mul_419 = None
        add_89: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(mul_416, mul_420);  mul_416 = mul_420 = None
        mul_421: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_411, add_89);  convert_element_type_411 = add_89 = None
        convert_element_type_413: "bf16[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(mul_421, torch.bfloat16);  mul_421 = None
        view_568: "bf16[25088, 2048]" = torch.ops.aten.view.default(convert_element_type_413, _shape_param_555);  convert_element_type_413 = _shape_param_555 = None
        mm_133: "bf16[25088, 512]" = torch.ops.aten.mm.default(view_568, arg472_1);  arg472_1 = None
        permute_179: "bf16[2048, 25088]" = torch.ops.aten.permute.default(view_568, [1, 0])
        mm_134: "bf16[2048, 512]" = torch.ops.aten.mm.default(permute_179, arg473_1);  permute_179 = arg473_1 = None
        sum_234: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_568, [0], True, dtype = torch.float32);  view_568 = None
        view_569: "f32[2048]" = torch.ops.aten.view.default(sum_234, _shape_param_556);  sum_234 = _shape_param_556 = None
        convert_element_type_414: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_569, torch.bfloat16);  view_569 = None
        view_570: "bf16[128, 196, 512]" = torch.ops.aten.view.default(mm_133, _shape_param_557);  mm_133 = _shape_param_557 = None
        convert_element_type_415: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(view_570, torch.float32);  view_570 = None
        convert_element_type_416: "f32[2048, 512]" = torch.ops.prims.convert_element_type.default(mm_134, torch.float32);  mm_134 = None
        convert_element_type_417: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_414, torch.float32);  convert_element_type_414 = None
        mul_422: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_415, arg474_1);  arg474_1 = None
        mul_423: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_422, 512)
        sum_235: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_422, [2], True)
        convert_element_type_418: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(arg475_1, torch.float32);  arg475_1 = None
        sub_118: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_418, arg476_1);  convert_element_type_418 = arg476_1 = None
        mul_424: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_118, arg477_1);  sub_118 = None
        mul_425: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_422, mul_424);  mul_422 = None
        sum_236: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_425, [2], True);  mul_425 = None
        mul_426: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_424, sum_236);  sum_236 = None
        sub_119: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(mul_423, sum_235);  mul_423 = sum_235 = None
        sub_120: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(sub_119, mul_426);  sub_119 = mul_426 = None
        div_84: "f32[128, 196, 1]" = torch.ops.aten.div.Tensor(arg477_1, 512);  arg477_1 = None
        mul_427: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(div_84, sub_120);  div_84 = sub_120 = None
        mul_428: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_415, mul_424);  mul_424 = None
        sum_237: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_428, [0, 1]);  mul_428 = None
        sum_238: "f32[512]" = torch.ops.aten.sum.dim_IntList(convert_element_type_415, [0, 1]);  convert_element_type_415 = None
        convert_element_type_419: "bf16[128, 196, 512]" = torch.ops.prims.convert_element_type.default(mul_427, torch.bfloat16);  mul_427 = None
        add_90: "bf16[128, 196, 512]" = torch.ops.aten.add.Tensor(view_563, convert_element_type_419);  view_563 = convert_element_type_419 = None
        view_571: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(add_90, _shape_param_558);  add_90 = _shape_param_558 = None
        convert_element_type_420: "bf16[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(arg478_1, torch.bfloat16);  arg478_1 = None
        div_85: "bf16[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_420, 0.9695652164518833);  convert_element_type_420 = None
        mul_429: "bf16[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_571, div_85);  div_85 = None
        index_44: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(mul_429, [None, None, fmod]);  mul_429 = None
        index_45: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_44, [None, fmod]);  index_44 = None
        view_572: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(index_45, _shape_param_559);  index_45 = _shape_param_559 = None
        permute_180: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_572, [0, 1, 3, 2, 4, 5]);  view_572 = None
        clone_77: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_180, memory_format = torch.contiguous_format);  permute_180 = None
        view_573: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(clone_77, _shape_param_560);  clone_77 = _shape_param_560 = None
        view_574: "bf16[512, 49, 512]" = torch.ops.aten.view.default(view_573, _shape_param_561);  view_573 = _shape_param_561 = None
        view_575: "bf16[25088, 512]" = torch.ops.aten.view.default(view_574, _shape_param_562);  view_574 = _shape_param_562 = None
        mm_135: "bf16[25088, 512]" = torch.ops.aten.mm.default(view_575, arg479_1);  arg479_1 = None
        permute_181: "bf16[512, 25088]" = torch.ops.aten.permute.default(view_575, [1, 0])
        mm_136: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_181, arg480_1);  permute_181 = arg480_1 = None
        sum_239: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_575, [0], True, dtype = torch.float32);  view_575 = None
        view_576: "f32[512]" = torch.ops.aten.view.default(sum_239, _shape_param_563);  sum_239 = _shape_param_563 = None
        convert_element_type_421: "bf16[512]" = torch.ops.prims.convert_element_type.default(view_576, torch.bfloat16);  view_576 = None
        view_577: "bf16[512, 49, 512]" = torch.ops.aten.view.default(mm_135, _shape_param_564);  mm_135 = _shape_param_564 = None
        convert_element_type_422: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_136, torch.float32);  mm_136 = None
        convert_element_type_423: "f32[512]" = torch.ops.prims.convert_element_type.default(convert_element_type_421, torch.float32);  convert_element_type_421 = None
        view_578: "bf16[512, 49, 16, 32]" = torch.ops.aten.view.default(view_577, _shape_param_565);  view_577 = _shape_param_565 = None
        permute_182: "bf16[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_578, [0, 2, 1, 3]);  view_578 = None
        clone_78: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(permute_182, memory_format = torch.contiguous_format);  permute_182 = None
        view_579: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_78, _shape_param_566);  clone_78 = _shape_param_566 = None
        bmm_64: "bf16[8192, 49, 32]" = torch.ops.aten.bmm.default(arg481_1, view_579);  arg481_1 = None
        bmm_65: "bf16[8192, 49, 49]" = torch.ops.aten.bmm.default(view_579, arg482_1);  view_579 = arg482_1 = None
        view_580: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(bmm_64, _shape_param_567);  bmm_64 = _shape_param_567 = None
        view_581: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(bmm_65, _shape_param_568);  bmm_65 = _shape_param_568 = None
        convert_element_type_424: "f32[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(view_581, torch.float32);  view_581 = None
        slice_29: "bf16[8192, 49, 56]" = torch.ops.aten.slice.Tensor(arg483_1, 1, 0, -7);  arg483_1 = None
        slice_30: "bf16[8192, 49, 49]" = torch.ops.aten.slice.Tensor(slice_29, 2, 0, -7);  slice_29 = None
        view_582: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(slice_30, _shape_param_569);  slice_30 = _shape_param_569 = None
        view_583: "i64[2401]" = torch.ops.aten.view.default(arg484_1, [-1]);  arg484_1 = None
        index_46: "f32[2401, 16]" = torch.ops.aten.index.Tensor(arg485_1, [view_583]);  arg485_1 = None
        view_584: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_46, _shape_param_570);  index_46 = _shape_param_570 = None
        permute_183: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_584, [2, 0, 1]);  view_584 = None
        clone_79: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_183, memory_format = torch.contiguous_format);  permute_183 = None
        unsqueeze_32: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_79, 0);  clone_79 = None
        add_91: "f32[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_582, unsqueeze_32);  view_582 = unsqueeze_32 = None
        view_585: "f32[128, 4, 16, 49, 49]" = torch.ops.aten.view.default(add_91, _shape_param_571);  add_91 = _shape_param_571 = None
        unsqueeze_33: "f32[4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(arg486_1, 1);  arg486_1 = None
        unsqueeze_34: "f32[1, 4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_33, 0);  unsqueeze_33 = None
        add_92: "f32[128, 4, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_585, unsqueeze_34);  view_585 = unsqueeze_34 = None
        view_586: "f32[512, 16, 49, 49]" = torch.ops.aten.view.default(add_92, _shape_param_572);  add_92 = _shape_param_572 = None
        sub_121: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(view_586, arg487_1);  view_586 = arg487_1 = None
        exp_33: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_121);  sub_121 = None
        div_86: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_33, arg488_1);  exp_33 = arg488_1 = None
        mul_430: "f32[512, 16, 49, 49]" = torch.ops.aten.mul.Tensor(convert_element_type_424, div_86);  convert_element_type_424 = None
        sum_240: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_430, [-1], True)
        neg_16: "f32[512, 16, 49, 49]" = torch.ops.aten.neg.default(div_86);  div_86 = None
        fma_16: "f32[512, 16, 49, 49]" = torch.ops.prims.fma.default(neg_16, sum_240, mul_430);  neg_16 = sum_240 = mul_430 = None
        convert_element_type_425: "bf16[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(fma_16, torch.bfloat16)
        sum_241: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_16, [0], True, dtype = torch.float32);  fma_16 = None
        squeeze_16: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_241, 0);  sum_241 = None
        permute_184: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_16, [1, 2, 0]);  squeeze_16 = None
        view_587: "f32[2401, 16]" = torch.ops.aten.view.default(permute_184, _shape_param_573);  permute_184 = _shape_param_573 = None
        index_put_16: "f32[169, 16]" = torch.ops.aten.index_put.default(full_1, [view_583], view_587, True);  view_583 = view_587 = None
        view_588: "bf16[8192, 49, 49]" = torch.ops.aten.view.default(convert_element_type_425, _shape_param_574);  convert_element_type_425 = _shape_param_574 = None
        bmm_66: "bf16[8192, 32, 49]" = torch.ops.aten.bmm.default(arg489_1, view_588);  arg489_1 = None
        bmm_67: "bf16[8192, 49, 32]" = torch.ops.aten.bmm.default(view_588, arg490_1);  view_588 = arg490_1 = None
        view_589: "bf16[512, 16, 32, 49]" = torch.ops.aten.view.default(bmm_66, _shape_param_575);  bmm_66 = _shape_param_575 = None
        view_590: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(bmm_67, _shape_param_576);  bmm_67 = _shape_param_576 = None
        permute_185: "bf16[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_589, [0, 1, 3, 2]);  view_589 = None
        mul_431: "bf16[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_590, 0.1767766952966369);  view_590 = None
        cat_16: "bf16[1536, 16, 49, 32]" = torch.ops.aten.cat.default([mul_431, permute_185, view_580]);  mul_431 = permute_185 = view_580 = None
        view_591: "bf16[3, 512, 16, 49, 32]" = torch.ops.aten.view.default(cat_16, _shape_param_577);  cat_16 = _shape_param_577 = None
        permute_186: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.permute.default(view_591, [1, 3, 0, 2, 4]);  view_591 = None
        clone_80: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_186, memory_format = torch.contiguous_format);  permute_186 = None
        view_592: "bf16[512, 49, 1536]" = torch.ops.aten.view.default(clone_80, _shape_param_578);  clone_80 = _shape_param_578 = None
        view_593: "bf16[25088, 1536]" = torch.ops.aten.view.default(view_592, _shape_param_579);  view_592 = _shape_param_579 = None
        mm_137: "bf16[25088, 512]" = torch.ops.aten.mm.default(view_593, arg491_1);  arg491_1 = None
        permute_187: "bf16[1536, 25088]" = torch.ops.aten.permute.default(view_593, [1, 0])
        mm_138: "bf16[1536, 512]" = torch.ops.aten.mm.default(permute_187, arg492_1);  permute_187 = arg492_1 = None
        sum_242: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_593, [0], True, dtype = torch.float32);  view_593 = None
        view_594: "f32[1536]" = torch.ops.aten.view.default(sum_242, _shape_param_580);  sum_242 = _shape_param_580 = None
        convert_element_type_426: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_594, torch.bfloat16);  view_594 = None
        view_595: "bf16[512, 49, 512]" = torch.ops.aten.view.default(mm_137, _shape_param_581);  mm_137 = _shape_param_581 = None
        convert_element_type_427: "f32[512, 49, 512]" = torch.ops.prims.convert_element_type.default(view_595, torch.float32);  view_595 = None
        convert_element_type_428: "f32[1536, 512]" = torch.ops.prims.convert_element_type.default(mm_138, torch.float32);  mm_138 = None
        convert_element_type_429: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_426, torch.float32);  convert_element_type_426 = None
        view_596: "f32[512, 7, 7, 512]" = torch.ops.aten.view.default(convert_element_type_427, _shape_param_582);  convert_element_type_427 = _shape_param_582 = None
        view_597: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_596, _shape_param_583);  view_596 = _shape_param_583 = None
        permute_188: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_597, [0, 1, 3, 2, 4, 5]);  view_597 = None
        clone_81: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_188, memory_format = torch.contiguous_format);  permute_188 = None
        view_598: "f32[128, 14, 14, 512]" = torch.ops.aten.view.default(clone_81, _shape_param_584);  clone_81 = _shape_param_584 = None
        index_47: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(view_598, [None, None, arg93_1]);  view_598 = None
        index_48: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_47, [None, arg93_1]);  index_47 = None
        mul_432: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(index_48, arg493_1);  arg493_1 = None
        mul_433: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_432, 512)
        sum_243: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_432, [3], True)
        convert_element_type_430: "f32[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(arg494_1, torch.float32);  arg494_1 = None
        sub_122: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_430, arg495_1);  convert_element_type_430 = arg495_1 = None
        mul_434: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_122, arg496_1);  sub_122 = None
        mul_435: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_432, mul_434);  mul_432 = None
        sum_244: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_435, [3], True);  mul_435 = None
        mul_436: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_434, sum_244);  sum_244 = None
        sub_123: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_433, sum_243);  mul_433 = sum_243 = None
        sub_124: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_123, mul_436);  sub_123 = mul_436 = None
        div_87: "f32[128, 14, 14, 1]" = torch.ops.aten.div.Tensor(arg496_1, 512);  arg496_1 = None
        mul_437: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_87, sub_124);  div_87 = sub_124 = None
        mul_438: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(index_48, mul_434);  mul_434 = None
        sum_245: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_438, [0, 1, 2]);  mul_438 = None
        sum_246: "f32[512]" = torch.ops.aten.sum.dim_IntList(index_48, [0, 1, 2]);  index_48 = None
        convert_element_type_431: "bf16[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(mul_437, torch.bfloat16);  mul_437 = None
        add_93: "bf16[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_571, convert_element_type_431);  view_571 = convert_element_type_431 = None
        view_599: "bf16[128, 196, 512]" = torch.ops.aten.view.default(add_93, _shape_param_585);  add_93 = _shape_param_585 = None
        convert_element_type_432: "bf16[128, 1, 1]" = torch.ops.prims.convert_element_type.default(arg497_1, torch.bfloat16);  arg497_1 = None
        div_88: "bf16[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_432, 0.9739130418747663);  convert_element_type_432 = None
        mul_439: "bf16[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_599, div_88);  div_88 = None
        view_600: "bf16[25088, 512]" = torch.ops.aten.view.default(mul_439, _shape_param_586);  mul_439 = _shape_param_586 = None
        mm_139: "bf16[25088, 2048]" = torch.ops.aten.mm.default(view_600, arg498_1);  arg498_1 = None
        permute_189: "bf16[512, 25088]" = torch.ops.aten.permute.default(view_600, [1, 0])
        mm_140: "bf16[512, 2048]" = torch.ops.aten.mm.default(permute_189, arg499_1);  permute_189 = arg499_1 = None
        sum_247: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_600, [0], True, dtype = torch.float32);  view_600 = None
        view_601: "f32[512]" = torch.ops.aten.view.default(sum_247, _shape_param_587);  sum_247 = _shape_param_587 = None
        convert_element_type_433: "bf16[512]" = torch.ops.prims.convert_element_type.default(view_601, torch.bfloat16);  view_601 = None
        view_602: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(mm_139, _shape_param_588);  mm_139 = _shape_param_588 = None
        convert_element_type_434: "f32[512, 2048]" = torch.ops.prims.convert_element_type.default(mm_140, torch.float32);  mm_140 = None
        convert_element_type_435: "f32[512]" = torch.ops.prims.convert_element_type.default(convert_element_type_433, torch.float32);  convert_element_type_433 = None
        convert_element_type_436: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_602, torch.float32);  view_602 = None
        view_603: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(arg500_1, _shape_param_589);  arg500_1 = _shape_param_589 = None
        convert_element_type_437: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_603, torch.float32);  view_603 = None
        mul_440: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_437, 0.7071067811865476)
        erf_17: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_440);  mul_440 = None
        add_94: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
        mul_441: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(add_94, 0.5);  add_94 = None
        mul_442: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_437, convert_element_type_437)
        mul_443: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_442, -0.5);  mul_442 = None
        exp_34: "f32[128, 196, 2048]" = torch.ops.aten.exp.default(mul_443);  mul_443 = None
        mul_444: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_34, 0.3989422804014327);  exp_34 = None
        mul_445: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_437, mul_444);  convert_element_type_437 = mul_444 = None
        add_95: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(mul_441, mul_445);  mul_441 = mul_445 = None
        mul_446: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_436, add_95);  convert_element_type_436 = add_95 = None
        convert_element_type_438: "bf16[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(mul_446, torch.bfloat16);  mul_446 = None
        view_604: "bf16[25088, 2048]" = torch.ops.aten.view.default(convert_element_type_438, _shape_param_590);  convert_element_type_438 = _shape_param_590 = None
        mm_141: "bf16[25088, 512]" = torch.ops.aten.mm.default(view_604, arg501_1);  arg501_1 = None
        permute_190: "bf16[2048, 25088]" = torch.ops.aten.permute.default(view_604, [1, 0])
        mm_142: "bf16[2048, 512]" = torch.ops.aten.mm.default(permute_190, arg502_1);  permute_190 = arg502_1 = None
        sum_248: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_604, [0], True, dtype = torch.float32);  view_604 = None
        view_605: "f32[2048]" = torch.ops.aten.view.default(sum_248, _shape_param_591);  sum_248 = _shape_param_591 = None
        convert_element_type_439: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_605, torch.bfloat16);  view_605 = None
        view_606: "bf16[128, 196, 512]" = torch.ops.aten.view.default(mm_141, _shape_param_592);  mm_141 = _shape_param_592 = None
        convert_element_type_440: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(view_606, torch.float32);  view_606 = None
        convert_element_type_441: "f32[2048, 512]" = torch.ops.prims.convert_element_type.default(mm_142, torch.float32);  mm_142 = None
        convert_element_type_442: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_439, torch.float32);  convert_element_type_439 = None
        mul_447: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_440, arg503_1);  arg503_1 = None
        mul_448: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_447, 512)
        sum_249: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_447, [2], True)
        convert_element_type_443: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(arg504_1, torch.float32);  arg504_1 = None
        sub_125: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_443, arg505_1);  convert_element_type_443 = arg505_1 = None
        mul_449: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_125, arg506_1);  sub_125 = None
        mul_450: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_447, mul_449);  mul_447 = None
        sum_250: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_450, [2], True);  mul_450 = None
        mul_451: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_449, sum_250);  sum_250 = None
        sub_126: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(mul_448, sum_249);  mul_448 = sum_249 = None
        sub_127: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(sub_126, mul_451);  sub_126 = mul_451 = None
        div_89: "f32[128, 196, 1]" = torch.ops.aten.div.Tensor(arg506_1, 512);  arg506_1 = None
        mul_452: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(div_89, sub_127);  div_89 = sub_127 = None
        mul_453: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_440, mul_449);  mul_449 = None
        sum_251: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_453, [0, 1]);  mul_453 = None
        sum_252: "f32[512]" = torch.ops.aten.sum.dim_IntList(convert_element_type_440, [0, 1]);  convert_element_type_440 = None
        convert_element_type_444: "bf16[128, 196, 512]" = torch.ops.prims.convert_element_type.default(mul_452, torch.bfloat16);  mul_452 = None
        add_96: "bf16[128, 196, 512]" = torch.ops.aten.add.Tensor(view_599, convert_element_type_444);  view_599 = convert_element_type_444 = None
        view_607: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(add_96, _shape_param_593);  add_96 = _shape_param_593 = None
        convert_element_type_445: "bf16[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(arg507_1, torch.bfloat16);  arg507_1 = None
        div_90: "bf16[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_445, 0.9739130418747663);  convert_element_type_445 = None
        mul_454: "bf16[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_607, div_90);  div_90 = None
        view_608: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(mul_454, _shape_param_594);  mul_454 = _shape_param_594 = None
        permute_191: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_608, [0, 1, 3, 2, 4, 5]);  view_608 = None
        clone_82: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_191, memory_format = torch.contiguous_format);  permute_191 = None
        view_609: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(clone_82, _shape_param_595);  clone_82 = _shape_param_595 = None
        view_610: "bf16[512, 49, 512]" = torch.ops.aten.view.default(view_609, _shape_param_596);  view_609 = _shape_param_596 = None
        view_611: "bf16[25088, 512]" = torch.ops.aten.view.default(view_610, _shape_param_597);  view_610 = _shape_param_597 = None
        mm_143: "bf16[25088, 512]" = torch.ops.aten.mm.default(view_611, arg508_1);  arg508_1 = None
        permute_192: "bf16[512, 25088]" = torch.ops.aten.permute.default(view_611, [1, 0])
        mm_144: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_192, arg509_1);  permute_192 = arg509_1 = None
        sum_253: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_611, [0], True, dtype = torch.float32);  view_611 = None
        view_612: "f32[512]" = torch.ops.aten.view.default(sum_253, _shape_param_598);  sum_253 = _shape_param_598 = None
        convert_element_type_446: "bf16[512]" = torch.ops.prims.convert_element_type.default(view_612, torch.bfloat16);  view_612 = None
        view_613: "bf16[512, 49, 512]" = torch.ops.aten.view.default(mm_143, _shape_param_599);  mm_143 = _shape_param_599 = None
        convert_element_type_447: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_144, torch.float32);  mm_144 = None
        convert_element_type_448: "f32[512]" = torch.ops.prims.convert_element_type.default(convert_element_type_446, torch.float32);  convert_element_type_446 = None
        view_614: "bf16[512, 49, 16, 32]" = torch.ops.aten.view.default(view_613, _shape_param_600);  view_613 = _shape_param_600 = None
        permute_193: "bf16[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_614, [0, 2, 1, 3]);  view_614 = None
        clone_83: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(permute_193, memory_format = torch.contiguous_format);  permute_193 = None
        view_615: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_83, _shape_param_601);  clone_83 = _shape_param_601 = None
        bmm_68: "bf16[8192, 49, 32]" = torch.ops.aten.bmm.default(arg510_1, view_615);  arg510_1 = None
        bmm_69: "bf16[8192, 49, 49]" = torch.ops.aten.bmm.default(view_615, arg511_1);  view_615 = arg511_1 = None
        view_616: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(bmm_68, _shape_param_602);  bmm_68 = _shape_param_602 = None
        view_617: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(bmm_69, _shape_param_603);  bmm_69 = _shape_param_603 = None
        convert_element_type_449: "f32[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(view_617, torch.float32);  view_617 = None
        slice_31: "bf16[8192, 49, 56]" = torch.ops.aten.slice.Tensor(arg512_1, 1, 0, -7);  arg512_1 = None
        slice_32: "bf16[8192, 49, 49]" = torch.ops.aten.slice.Tensor(slice_31, 2, 0, -7);  slice_31 = None
        view_618: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(slice_32, _shape_param_604);  slice_32 = _shape_param_604 = None
        view_619: "i64[2401]" = torch.ops.aten.view.default(arg513_1, [-1]);  arg513_1 = None
        index_49: "f32[2401, 16]" = torch.ops.aten.index.Tensor(arg514_1, [view_619]);  arg514_1 = None
        view_620: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_49, _shape_param_605);  index_49 = _shape_param_605 = None
        permute_194: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_620, [2, 0, 1]);  view_620 = None
        clone_84: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_194, memory_format = torch.contiguous_format);  permute_194 = None
        unsqueeze_35: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_84, 0);  clone_84 = None
        add_97: "f32[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_618, unsqueeze_35);  view_618 = unsqueeze_35 = None
        sub_128: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(add_97, arg515_1);  add_97 = arg515_1 = None
        exp_35: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_128);  sub_128 = None
        div_91: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_35, arg516_1);  exp_35 = arg516_1 = None
        mul_455: "f32[512, 16, 49, 49]" = torch.ops.aten.mul.Tensor(convert_element_type_449, div_91);  convert_element_type_449 = None
        sum_254: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_455, [-1], True)
        neg_17: "f32[512, 16, 49, 49]" = torch.ops.aten.neg.default(div_91);  div_91 = None
        fma_17: "f32[512, 16, 49, 49]" = torch.ops.prims.fma.default(neg_17, sum_254, mul_455);  neg_17 = sum_254 = mul_455 = None
        convert_element_type_450: "bf16[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(fma_17, torch.bfloat16)
        sum_255: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_17, [0], True, dtype = torch.float32);  fma_17 = None
        squeeze_17: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_255, 0);  sum_255 = None
        permute_195: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_17, [1, 2, 0]);  squeeze_17 = None
        view_621: "f32[2401, 16]" = torch.ops.aten.view.default(permute_195, _shape_param_606);  permute_195 = _shape_param_606 = None
        index_put_17: "f32[169, 16]" = torch.ops.aten.index_put.default(full_1, [view_619], view_621, True);  view_619 = view_621 = None
        view_622: "bf16[8192, 49, 49]" = torch.ops.aten.view.default(convert_element_type_450, _shape_param_607);  convert_element_type_450 = _shape_param_607 = None
        bmm_70: "bf16[8192, 32, 49]" = torch.ops.aten.bmm.default(arg517_1, view_622);  arg517_1 = None
        bmm_71: "bf16[8192, 49, 32]" = torch.ops.aten.bmm.default(view_622, arg518_1);  view_622 = arg518_1 = None
        view_623: "bf16[512, 16, 32, 49]" = torch.ops.aten.view.default(bmm_70, _shape_param_608);  bmm_70 = _shape_param_608 = None
        view_624: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(bmm_71, _shape_param_609);  bmm_71 = _shape_param_609 = None
        permute_196: "bf16[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_623, [0, 1, 3, 2]);  view_623 = None
        mul_456: "bf16[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_624, 0.1767766952966369);  view_624 = None
        cat_17: "bf16[1536, 16, 49, 32]" = torch.ops.aten.cat.default([mul_456, permute_196, view_616]);  mul_456 = permute_196 = view_616 = None
        view_625: "bf16[3, 512, 16, 49, 32]" = torch.ops.aten.view.default(cat_17, _shape_param_610);  cat_17 = _shape_param_610 = None
        permute_197: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.permute.default(view_625, [1, 3, 0, 2, 4]);  view_625 = None
        clone_85: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_197, memory_format = torch.contiguous_format);  permute_197 = None
        view_626: "bf16[512, 49, 1536]" = torch.ops.aten.view.default(clone_85, _shape_param_611);  clone_85 = _shape_param_611 = None
        view_627: "bf16[25088, 1536]" = torch.ops.aten.view.default(view_626, _shape_param_612);  view_626 = _shape_param_612 = None
        mm_145: "bf16[25088, 512]" = torch.ops.aten.mm.default(view_627, arg519_1);  arg519_1 = None
        permute_198: "bf16[1536, 25088]" = torch.ops.aten.permute.default(view_627, [1, 0])
        mm_146: "bf16[1536, 512]" = torch.ops.aten.mm.default(permute_198, arg520_1);  permute_198 = arg520_1 = None
        sum_256: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_627, [0], True, dtype = torch.float32);  view_627 = None
        view_628: "f32[1536]" = torch.ops.aten.view.default(sum_256, _shape_param_613);  sum_256 = _shape_param_613 = None
        convert_element_type_451: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_628, torch.bfloat16);  view_628 = None
        view_629: "bf16[512, 49, 512]" = torch.ops.aten.view.default(mm_145, _shape_param_614);  mm_145 = _shape_param_614 = None
        convert_element_type_452: "f32[512, 49, 512]" = torch.ops.prims.convert_element_type.default(view_629, torch.float32);  view_629 = None
        convert_element_type_453: "f32[1536, 512]" = torch.ops.prims.convert_element_type.default(mm_146, torch.float32);  mm_146 = None
        convert_element_type_454: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_451, torch.float32);  convert_element_type_451 = None
        view_630: "f32[512, 7, 7, 512]" = torch.ops.aten.view.default(convert_element_type_452, _shape_param_615);  convert_element_type_452 = _shape_param_615 = None
        view_631: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_630, _shape_param_616);  view_630 = _shape_param_616 = None
        permute_199: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_631, [0, 1, 3, 2, 4, 5]);  view_631 = None
        clone_86: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_199, memory_format = torch.contiguous_format);  permute_199 = None
        view_632: "f32[128, 14, 14, 512]" = torch.ops.aten.view.default(clone_86, _shape_param_617);  clone_86 = _shape_param_617 = None
        mul_457: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_632, arg521_1);  arg521_1 = None
        mul_458: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_457, 512)
        sum_257: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_457, [3], True)
        convert_element_type_455: "f32[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(arg522_1, torch.float32);  arg522_1 = None
        sub_129: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_455, arg523_1);  convert_element_type_455 = arg523_1 = None
        mul_459: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_129, arg524_1);  sub_129 = None
        mul_460: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_457, mul_459);  mul_457 = None
        sum_258: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_460, [3], True);  mul_460 = None
        mul_461: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_459, sum_258);  sum_258 = None
        sub_130: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_458, sum_257);  mul_458 = sum_257 = None
        sub_131: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_130, mul_461);  sub_130 = mul_461 = None
        div_92: "f32[128, 14, 14, 1]" = torch.ops.aten.div.Tensor(arg524_1, 512);  arg524_1 = None
        mul_462: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_92, sub_131);  div_92 = sub_131 = None
        mul_463: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_632, mul_459);  mul_459 = None
        sum_259: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_463, [0, 1, 2]);  mul_463 = None
        sum_260: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_632, [0, 1, 2]);  view_632 = None
        convert_element_type_456: "bf16[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(mul_462, torch.bfloat16);  mul_462 = None
        add_98: "bf16[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_607, convert_element_type_456);  view_607 = convert_element_type_456 = None
        view_633: "bf16[128, 196, 512]" = torch.ops.aten.view.default(add_98, _shape_param_618);  add_98 = _shape_param_618 = None
        convert_element_type_457: "bf16[128, 1, 1]" = torch.ops.prims.convert_element_type.default(arg525_1, torch.bfloat16);  arg525_1 = None
        div_93: "bf16[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_457, 0.9782608672976494);  convert_element_type_457 = None
        mul_464: "bf16[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_633, div_93);  div_93 = None
        view_634: "bf16[25088, 512]" = torch.ops.aten.view.default(mul_464, _shape_param_619);  mul_464 = _shape_param_619 = None
        mm_147: "bf16[25088, 2048]" = torch.ops.aten.mm.default(view_634, arg526_1);  arg526_1 = None
        permute_200: "bf16[512, 25088]" = torch.ops.aten.permute.default(view_634, [1, 0])
        mm_148: "bf16[512, 2048]" = torch.ops.aten.mm.default(permute_200, arg527_1);  permute_200 = arg527_1 = None
        sum_261: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_634, [0], True, dtype = torch.float32);  view_634 = None
        view_635: "f32[512]" = torch.ops.aten.view.default(sum_261, _shape_param_620);  sum_261 = _shape_param_620 = None
        convert_element_type_458: "bf16[512]" = torch.ops.prims.convert_element_type.default(view_635, torch.bfloat16);  view_635 = None
        view_636: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(mm_147, _shape_param_621);  mm_147 = _shape_param_621 = None
        convert_element_type_459: "f32[512, 2048]" = torch.ops.prims.convert_element_type.default(mm_148, torch.float32);  mm_148 = None
        convert_element_type_460: "f32[512]" = torch.ops.prims.convert_element_type.default(convert_element_type_458, torch.float32);  convert_element_type_458 = None
        convert_element_type_461: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_636, torch.float32);  view_636 = None
        view_637: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(arg528_1, _shape_param_622);  arg528_1 = _shape_param_622 = None
        convert_element_type_462: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_637, torch.float32);  view_637 = None
        mul_465: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_462, 0.7071067811865476)
        erf_18: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_465);  mul_465 = None
        add_99: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
        mul_466: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(add_99, 0.5);  add_99 = None
        mul_467: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_462, convert_element_type_462)
        mul_468: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_467, -0.5);  mul_467 = None
        exp_36: "f32[128, 196, 2048]" = torch.ops.aten.exp.default(mul_468);  mul_468 = None
        mul_469: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_36, 0.3989422804014327);  exp_36 = None
        mul_470: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_462, mul_469);  convert_element_type_462 = mul_469 = None
        add_100: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(mul_466, mul_470);  mul_466 = mul_470 = None
        mul_471: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_461, add_100);  convert_element_type_461 = add_100 = None
        convert_element_type_463: "bf16[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(mul_471, torch.bfloat16);  mul_471 = None
        view_638: "bf16[25088, 2048]" = torch.ops.aten.view.default(convert_element_type_463, _shape_param_623);  convert_element_type_463 = _shape_param_623 = None
        mm_149: "bf16[25088, 512]" = torch.ops.aten.mm.default(view_638, arg529_1);  arg529_1 = None
        permute_201: "bf16[2048, 25088]" = torch.ops.aten.permute.default(view_638, [1, 0])
        mm_150: "bf16[2048, 512]" = torch.ops.aten.mm.default(permute_201, arg530_1);  permute_201 = arg530_1 = None
        sum_262: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_638, [0], True, dtype = torch.float32);  view_638 = None
        view_639: "f32[2048]" = torch.ops.aten.view.default(sum_262, _shape_param_624);  sum_262 = _shape_param_624 = None
        convert_element_type_464: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_639, torch.bfloat16);  view_639 = None
        view_640: "bf16[128, 196, 512]" = torch.ops.aten.view.default(mm_149, _shape_param_625);  mm_149 = _shape_param_625 = None
        convert_element_type_465: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(view_640, torch.float32);  view_640 = None
        convert_element_type_466: "f32[2048, 512]" = torch.ops.prims.convert_element_type.default(mm_150, torch.float32);  mm_150 = None
        convert_element_type_467: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_464, torch.float32);  convert_element_type_464 = None
        mul_472: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_465, arg531_1);  arg531_1 = None
        mul_473: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_472, 512)
        sum_263: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_472, [2], True)
        convert_element_type_468: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(arg532_1, torch.float32);  arg532_1 = None
        sub_132: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_468, arg533_1);  convert_element_type_468 = arg533_1 = None
        mul_474: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_132, arg534_1);  sub_132 = None
        mul_475: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_472, mul_474);  mul_472 = None
        sum_264: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_475, [2], True);  mul_475 = None
        mul_476: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_474, sum_264);  sum_264 = None
        sub_133: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(mul_473, sum_263);  mul_473 = sum_263 = None
        sub_134: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(sub_133, mul_476);  sub_133 = mul_476 = None
        div_94: "f32[128, 196, 1]" = torch.ops.aten.div.Tensor(arg534_1, 512);  arg534_1 = None
        mul_477: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(div_94, sub_134);  div_94 = sub_134 = None
        mul_478: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_465, mul_474);  mul_474 = None
        sum_265: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_478, [0, 1]);  mul_478 = None
        sum_266: "f32[512]" = torch.ops.aten.sum.dim_IntList(convert_element_type_465, [0, 1]);  convert_element_type_465 = None
        convert_element_type_469: "bf16[128, 196, 512]" = torch.ops.prims.convert_element_type.default(mul_477, torch.bfloat16);  mul_477 = None
        add_101: "bf16[128, 196, 512]" = torch.ops.aten.add.Tensor(view_633, convert_element_type_469);  view_633 = convert_element_type_469 = None
        view_641: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(add_101, _shape_param_626);  add_101 = _shape_param_626 = None
        convert_element_type_470: "bf16[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(arg535_1, torch.bfloat16);  arg535_1 = None
        div_95: "bf16[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_470, 0.9782608672976494);  convert_element_type_470 = None
        mul_479: "bf16[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_641, div_95);  div_95 = None
        index_50: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(mul_479, [None, None, fmod]);  mul_479 = None
        index_51: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_50, [None, fmod]);  index_50 = fmod = None
        view_642: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(index_51, _shape_param_627);  index_51 = _shape_param_627 = None
        permute_202: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_642, [0, 1, 3, 2, 4, 5]);  view_642 = None
        clone_87: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_202, memory_format = torch.contiguous_format);  permute_202 = None
        view_643: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(clone_87, _shape_param_628);  clone_87 = _shape_param_628 = None
        view_644: "bf16[512, 49, 512]" = torch.ops.aten.view.default(view_643, _shape_param_629);  view_643 = _shape_param_629 = None
        view_645: "bf16[25088, 512]" = torch.ops.aten.view.default(view_644, _shape_param_630);  view_644 = _shape_param_630 = None
        mm_151: "bf16[25088, 512]" = torch.ops.aten.mm.default(view_645, arg536_1);  arg536_1 = None
        permute_203: "bf16[512, 25088]" = torch.ops.aten.permute.default(view_645, [1, 0])
        mm_152: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_203, arg537_1);  permute_203 = arg537_1 = None
        sum_267: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_645, [0], True, dtype = torch.float32);  view_645 = None
        view_646: "f32[512]" = torch.ops.aten.view.default(sum_267, _shape_param_631);  sum_267 = _shape_param_631 = None
        convert_element_type_471: "bf16[512]" = torch.ops.prims.convert_element_type.default(view_646, torch.bfloat16);  view_646 = None
        view_647: "bf16[512, 49, 512]" = torch.ops.aten.view.default(mm_151, _shape_param_632);  mm_151 = _shape_param_632 = None
        convert_element_type_472: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_152, torch.float32);  mm_152 = None
        convert_element_type_473: "f32[512]" = torch.ops.prims.convert_element_type.default(convert_element_type_471, torch.float32);  convert_element_type_471 = None
        view_648: "bf16[512, 49, 16, 32]" = torch.ops.aten.view.default(view_647, _shape_param_633);  view_647 = _shape_param_633 = None
        permute_204: "bf16[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_648, [0, 2, 1, 3]);  view_648 = None
        clone_88: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(permute_204, memory_format = torch.contiguous_format);  permute_204 = None
        view_649: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_88, _shape_param_634);  clone_88 = _shape_param_634 = None
        bmm_72: "bf16[8192, 49, 32]" = torch.ops.aten.bmm.default(arg538_1, view_649);  arg538_1 = None
        bmm_73: "bf16[8192, 49, 49]" = torch.ops.aten.bmm.default(view_649, arg539_1);  view_649 = arg539_1 = None
        view_650: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(bmm_72, _shape_param_635);  bmm_72 = _shape_param_635 = None
        view_651: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(bmm_73, _shape_param_636);  bmm_73 = _shape_param_636 = None
        convert_element_type_474: "f32[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(view_651, torch.float32);  view_651 = None
        slice_33: "bf16[8192, 49, 56]" = torch.ops.aten.slice.Tensor(arg540_1, 1, 0, -7);  arg540_1 = None
        slice_34: "bf16[8192, 49, 49]" = torch.ops.aten.slice.Tensor(slice_33, 2, 0, -7);  slice_33 = None
        view_652: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(slice_34, _shape_param_637);  slice_34 = _shape_param_637 = None
        view_653: "i64[2401]" = torch.ops.aten.view.default(arg541_1, [-1]);  arg541_1 = None
        index_52: "f32[2401, 16]" = torch.ops.aten.index.Tensor(arg542_1, [view_653]);  arg542_1 = None
        view_654: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_52, _shape_param_638);  index_52 = _shape_param_638 = None
        permute_205: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_654, [2, 0, 1]);  view_654 = None
        clone_89: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_205, memory_format = torch.contiguous_format);  permute_205 = None
        unsqueeze_36: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_89, 0);  clone_89 = None
        add_102: "f32[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_652, unsqueeze_36);  view_652 = unsqueeze_36 = None
        view_655: "f32[128, 4, 16, 49, 49]" = torch.ops.aten.view.default(add_102, _shape_param_639);  add_102 = _shape_param_639 = None
        unsqueeze_37: "f32[4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(arg543_1, 1);  arg543_1 = None
        unsqueeze_38: "f32[1, 4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_37, 0);  unsqueeze_37 = None
        add_103: "f32[128, 4, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_655, unsqueeze_38);  view_655 = unsqueeze_38 = None
        view_656: "f32[512, 16, 49, 49]" = torch.ops.aten.view.default(add_103, _shape_param_640);  add_103 = _shape_param_640 = None
        sub_135: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(view_656, arg544_1);  view_656 = arg544_1 = None
        exp_37: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_135);  sub_135 = None
        div_96: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_37, arg545_1);  exp_37 = arg545_1 = None
        mul_480: "f32[512, 16, 49, 49]" = torch.ops.aten.mul.Tensor(convert_element_type_474, div_96);  convert_element_type_474 = None
        sum_268: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_480, [-1], True)
        neg_18: "f32[512, 16, 49, 49]" = torch.ops.aten.neg.default(div_96);  div_96 = None
        fma_18: "f32[512, 16, 49, 49]" = torch.ops.prims.fma.default(neg_18, sum_268, mul_480);  neg_18 = sum_268 = mul_480 = None
        convert_element_type_475: "bf16[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(fma_18, torch.bfloat16)
        sum_269: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_18, [0], True, dtype = torch.float32);  fma_18 = None
        squeeze_18: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_269, 0);  sum_269 = None
        permute_206: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_18, [1, 2, 0]);  squeeze_18 = None
        view_657: "f32[2401, 16]" = torch.ops.aten.view.default(permute_206, _shape_param_641);  permute_206 = _shape_param_641 = None
        index_put_18: "f32[169, 16]" = torch.ops.aten.index_put.default(full_1, [view_653], view_657, True);  view_653 = view_657 = None
        view_658: "bf16[8192, 49, 49]" = torch.ops.aten.view.default(convert_element_type_475, _shape_param_642);  convert_element_type_475 = _shape_param_642 = None
        bmm_74: "bf16[8192, 32, 49]" = torch.ops.aten.bmm.default(arg546_1, view_658);  arg546_1 = None
        bmm_75: "bf16[8192, 49, 32]" = torch.ops.aten.bmm.default(view_658, arg547_1);  view_658 = arg547_1 = None
        view_659: "bf16[512, 16, 32, 49]" = torch.ops.aten.view.default(bmm_74, _shape_param_643);  bmm_74 = _shape_param_643 = None
        view_660: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(bmm_75, _shape_param_644);  bmm_75 = _shape_param_644 = None
        permute_207: "bf16[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_659, [0, 1, 3, 2]);  view_659 = None
        mul_481: "bf16[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_660, 0.1767766952966369);  view_660 = None
        cat_18: "bf16[1536, 16, 49, 32]" = torch.ops.aten.cat.default([mul_481, permute_207, view_650]);  mul_481 = permute_207 = view_650 = None
        view_661: "bf16[3, 512, 16, 49, 32]" = torch.ops.aten.view.default(cat_18, _shape_param_645);  cat_18 = _shape_param_645 = None
        permute_208: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.permute.default(view_661, [1, 3, 0, 2, 4]);  view_661 = None
        clone_90: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_208, memory_format = torch.contiguous_format);  permute_208 = None
        view_662: "bf16[512, 49, 1536]" = torch.ops.aten.view.default(clone_90, _shape_param_646);  clone_90 = _shape_param_646 = None
        view_663: "bf16[25088, 1536]" = torch.ops.aten.view.default(view_662, _shape_param_647);  view_662 = _shape_param_647 = None
        mm_153: "bf16[25088, 512]" = torch.ops.aten.mm.default(view_663, arg548_1);  arg548_1 = None
        permute_209: "bf16[1536, 25088]" = torch.ops.aten.permute.default(view_663, [1, 0])
        mm_154: "bf16[1536, 512]" = torch.ops.aten.mm.default(permute_209, arg549_1);  permute_209 = arg549_1 = None
        sum_270: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_663, [0], True, dtype = torch.float32);  view_663 = None
        view_664: "f32[1536]" = torch.ops.aten.view.default(sum_270, _shape_param_648);  sum_270 = _shape_param_648 = None
        convert_element_type_476: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_664, torch.bfloat16);  view_664 = None
        view_665: "bf16[512, 49, 512]" = torch.ops.aten.view.default(mm_153, _shape_param_649);  mm_153 = _shape_param_649 = None
        convert_element_type_477: "f32[512, 49, 512]" = torch.ops.prims.convert_element_type.default(view_665, torch.float32);  view_665 = None
        convert_element_type_478: "f32[1536, 512]" = torch.ops.prims.convert_element_type.default(mm_154, torch.float32);  mm_154 = None
        convert_element_type_479: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_476, torch.float32);  convert_element_type_476 = None
        view_666: "f32[512, 7, 7, 512]" = torch.ops.aten.view.default(convert_element_type_477, _shape_param_650);  convert_element_type_477 = _shape_param_650 = None
        view_667: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_666, _shape_param_651);  view_666 = _shape_param_651 = None
        permute_210: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_667, [0, 1, 3, 2, 4, 5]);  view_667 = None
        clone_91: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_210, memory_format = torch.contiguous_format);  permute_210 = None
        view_668: "f32[128, 14, 14, 512]" = torch.ops.aten.view.default(clone_91, _shape_param_652);  clone_91 = _shape_param_652 = None
        index_53: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(view_668, [None, None, arg93_1]);  view_668 = None
        index_54: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_53, [None, arg93_1]);  index_53 = arg93_1 = None
        mul_482: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(index_54, arg550_1);  arg550_1 = None
        mul_483: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_482, 512)
        sum_271: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_482, [3], True)
        convert_element_type_480: "f32[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(arg551_1, torch.float32);  arg551_1 = None
        sub_136: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_480, arg552_1);  convert_element_type_480 = arg552_1 = None
        mul_484: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_136, arg553_1);  sub_136 = None
        mul_485: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_482, mul_484);  mul_482 = None
        sum_272: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_485, [3], True);  mul_485 = None
        mul_486: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_484, sum_272);  sum_272 = None
        sub_137: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_483, sum_271);  mul_483 = sum_271 = None
        sub_138: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_137, mul_486);  sub_137 = mul_486 = None
        div_97: "f32[128, 14, 14, 1]" = torch.ops.aten.div.Tensor(arg553_1, 512);  arg553_1 = None
        mul_487: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_97, sub_138);  div_97 = sub_138 = None
        mul_488: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(index_54, mul_484);  mul_484 = None
        sum_273: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_488, [0, 1, 2]);  mul_488 = None
        sum_274: "f32[512]" = torch.ops.aten.sum.dim_IntList(index_54, [0, 1, 2]);  index_54 = None
        convert_element_type_481: "bf16[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(mul_487, torch.bfloat16);  mul_487 = None
        add_104: "bf16[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_641, convert_element_type_481);  view_641 = convert_element_type_481 = None
        view_669: "bf16[128, 196, 512]" = torch.ops.aten.view.default(add_104, _shape_param_653);  add_104 = _shape_param_653 = None
        convert_element_type_482: "bf16[128, 1, 1]" = torch.ops.prims.convert_element_type.default(arg554_1, torch.bfloat16);  arg554_1 = None
        div_98: "bf16[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_482, 0.9826086945831776);  convert_element_type_482 = None
        mul_489: "bf16[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_669, div_98);  div_98 = None
        view_670: "bf16[25088, 512]" = torch.ops.aten.view.default(mul_489, _shape_param_654);  mul_489 = _shape_param_654 = None
        mm_155: "bf16[25088, 2048]" = torch.ops.aten.mm.default(view_670, arg555_1);  arg555_1 = None
        permute_211: "bf16[512, 25088]" = torch.ops.aten.permute.default(view_670, [1, 0])
        mm_156: "bf16[512, 2048]" = torch.ops.aten.mm.default(permute_211, arg556_1);  permute_211 = arg556_1 = None
        sum_275: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_670, [0], True, dtype = torch.float32);  view_670 = None
        view_671: "f32[512]" = torch.ops.aten.view.default(sum_275, _shape_param_655);  sum_275 = _shape_param_655 = None
        convert_element_type_483: "bf16[512]" = torch.ops.prims.convert_element_type.default(view_671, torch.bfloat16);  view_671 = None
        view_672: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(mm_155, _shape_param_656);  mm_155 = _shape_param_656 = None
        convert_element_type_484: "f32[512, 2048]" = torch.ops.prims.convert_element_type.default(mm_156, torch.float32);  mm_156 = None
        convert_element_type_485: "f32[512]" = torch.ops.prims.convert_element_type.default(convert_element_type_483, torch.float32);  convert_element_type_483 = None
        convert_element_type_486: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_672, torch.float32);  view_672 = None
        view_673: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(arg557_1, _shape_param_657);  arg557_1 = _shape_param_657 = None
        convert_element_type_487: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_673, torch.float32);  view_673 = None
        mul_490: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_487, 0.7071067811865476)
        erf_19: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_490);  mul_490 = None
        add_105: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
        mul_491: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(add_105, 0.5);  add_105 = None
        mul_492: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_487, convert_element_type_487)
        mul_493: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_492, -0.5);  mul_492 = None
        exp_38: "f32[128, 196, 2048]" = torch.ops.aten.exp.default(mul_493);  mul_493 = None
        mul_494: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_38, 0.3989422804014327);  exp_38 = None
        mul_495: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_487, mul_494);  convert_element_type_487 = mul_494 = None
        add_106: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(mul_491, mul_495);  mul_491 = mul_495 = None
        mul_496: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_486, add_106);  convert_element_type_486 = add_106 = None
        convert_element_type_488: "bf16[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(mul_496, torch.bfloat16);  mul_496 = None
        view_674: "bf16[25088, 2048]" = torch.ops.aten.view.default(convert_element_type_488, _shape_param_658);  convert_element_type_488 = _shape_param_658 = None
        mm_157: "bf16[25088, 512]" = torch.ops.aten.mm.default(view_674, arg558_1);  arg558_1 = None
        permute_212: "bf16[2048, 25088]" = torch.ops.aten.permute.default(view_674, [1, 0])
        mm_158: "bf16[2048, 512]" = torch.ops.aten.mm.default(permute_212, arg559_1);  permute_212 = arg559_1 = None
        sum_276: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_674, [0], True, dtype = torch.float32);  view_674 = None
        view_675: "f32[2048]" = torch.ops.aten.view.default(sum_276, _shape_param_659);  sum_276 = _shape_param_659 = None
        convert_element_type_489: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_675, torch.bfloat16);  view_675 = None
        view_676: "bf16[128, 196, 512]" = torch.ops.aten.view.default(mm_157, _shape_param_660);  mm_157 = _shape_param_660 = None
        convert_element_type_490: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(view_676, torch.float32);  view_676 = None
        convert_element_type_491: "f32[2048, 512]" = torch.ops.prims.convert_element_type.default(mm_158, torch.float32);  mm_158 = None
        convert_element_type_492: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_489, torch.float32);  convert_element_type_489 = None
        mul_497: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_490, arg560_1);  arg560_1 = None
        mul_498: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_497, 512)
        sum_277: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_497, [2], True)
        convert_element_type_493: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(arg561_1, torch.float32);  arg561_1 = None
        sub_139: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_493, arg562_1);  convert_element_type_493 = arg562_1 = None
        mul_499: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_139, arg563_1);  sub_139 = None
        mul_500: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_497, mul_499);  mul_497 = None
        sum_278: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_500, [2], True);  mul_500 = None
        mul_501: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_499, sum_278);  sum_278 = None
        sub_140: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(mul_498, sum_277);  mul_498 = sum_277 = None
        sub_141: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(sub_140, mul_501);  sub_140 = mul_501 = None
        div_99: "f32[128, 196, 1]" = torch.ops.aten.div.Tensor(arg563_1, 512);  arg563_1 = None
        mul_502: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(div_99, sub_141);  div_99 = sub_141 = None
        mul_503: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_490, mul_499);  mul_499 = None
        sum_279: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_503, [0, 1]);  mul_503 = None
        sum_280: "f32[512]" = torch.ops.aten.sum.dim_IntList(convert_element_type_490, [0, 1]);  convert_element_type_490 = None
        convert_element_type_494: "bf16[128, 196, 512]" = torch.ops.prims.convert_element_type.default(mul_502, torch.bfloat16);  mul_502 = None
        add_107: "bf16[128, 196, 512]" = torch.ops.aten.add.Tensor(view_669, convert_element_type_494);  view_669 = convert_element_type_494 = None
        view_677: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(add_107, _shape_param_661);  add_107 = _shape_param_661 = None
        convert_element_type_495: "bf16[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(arg564_1, torch.bfloat16);  arg564_1 = None
        div_100: "bf16[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_495, 0.9826086945831776);  convert_element_type_495 = None
        mul_504: "bf16[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_677, div_100);  div_100 = None
        view_678: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(mul_504, _shape_param_662);  mul_504 = _shape_param_662 = None
        permute_213: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_678, [0, 1, 3, 2, 4, 5]);  view_678 = None
        clone_92: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_213, memory_format = torch.contiguous_format);  permute_213 = None
        view_679: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(clone_92, _shape_param_663);  clone_92 = _shape_param_663 = None
        view_680: "bf16[512, 49, 512]" = torch.ops.aten.view.default(view_679, _shape_param_664);  view_679 = _shape_param_664 = None
        view_681: "bf16[25088, 512]" = torch.ops.aten.view.default(view_680, _shape_param_665);  view_680 = _shape_param_665 = None
        mm_159: "bf16[25088, 512]" = torch.ops.aten.mm.default(view_681, arg565_1);  arg565_1 = None
        permute_214: "bf16[512, 25088]" = torch.ops.aten.permute.default(view_681, [1, 0])
        mm_160: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_214, arg566_1);  permute_214 = arg566_1 = None
        sum_281: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_681, [0], True, dtype = torch.float32);  view_681 = None
        view_682: "f32[512]" = torch.ops.aten.view.default(sum_281, _shape_param_666);  sum_281 = _shape_param_666 = None
        convert_element_type_496: "bf16[512]" = torch.ops.prims.convert_element_type.default(view_682, torch.bfloat16);  view_682 = None
        view_683: "bf16[512, 49, 512]" = torch.ops.aten.view.default(mm_159, _shape_param_667);  mm_159 = _shape_param_667 = None
        convert_element_type_497: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_160, torch.float32);  mm_160 = None
        convert_element_type_498: "f32[512]" = torch.ops.prims.convert_element_type.default(convert_element_type_496, torch.float32);  convert_element_type_496 = None
        view_684: "bf16[512, 49, 16, 32]" = torch.ops.aten.view.default(view_683, _shape_param_668);  view_683 = _shape_param_668 = None
        permute_215: "bf16[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_684, [0, 2, 1, 3]);  view_684 = None
        clone_93: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(permute_215, memory_format = torch.contiguous_format);  permute_215 = None
        view_685: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_93, _shape_param_669);  clone_93 = _shape_param_669 = None
        bmm_76: "bf16[8192, 49, 32]" = torch.ops.aten.bmm.default(arg567_1, view_685);  arg567_1 = None
        bmm_77: "bf16[8192, 49, 49]" = torch.ops.aten.bmm.default(view_685, arg568_1);  view_685 = arg568_1 = None
        view_686: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(bmm_76, _shape_param_670);  bmm_76 = _shape_param_670 = None
        view_687: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(bmm_77, _shape_param_671);  bmm_77 = _shape_param_671 = None
        convert_element_type_499: "f32[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(view_687, torch.float32);  view_687 = None
        slice_35: "bf16[8192, 49, 56]" = torch.ops.aten.slice.Tensor(arg569_1, 1, 0, -7);  arg569_1 = None
        slice_36: "bf16[8192, 49, 49]" = torch.ops.aten.slice.Tensor(slice_35, 2, 0, -7);  slice_35 = None
        view_688: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(slice_36, _shape_param_672);  slice_36 = _shape_param_672 = None
        view_689: "i64[2401]" = torch.ops.aten.view.default(arg570_1, [-1]);  arg570_1 = None
        index_55: "f32[2401, 16]" = torch.ops.aten.index.Tensor(arg571_1, [view_689]);  arg571_1 = None
        view_690: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_55, _shape_param_673);  index_55 = _shape_param_673 = None
        permute_216: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_690, [2, 0, 1]);  view_690 = None
        clone_94: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_216, memory_format = torch.contiguous_format);  permute_216 = None
        unsqueeze_39: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_94, 0);  clone_94 = None
        add_108: "f32[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_688, unsqueeze_39);  view_688 = unsqueeze_39 = None
        sub_142: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(add_108, arg572_1);  add_108 = arg572_1 = None
        exp_39: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_142);  sub_142 = None
        div_101: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_39, arg573_1);  exp_39 = arg573_1 = None
        mul_505: "f32[512, 16, 49, 49]" = torch.ops.aten.mul.Tensor(convert_element_type_499, div_101);  convert_element_type_499 = None
        sum_282: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_505, [-1], True)
        neg_19: "f32[512, 16, 49, 49]" = torch.ops.aten.neg.default(div_101);  div_101 = None
        fma_19: "f32[512, 16, 49, 49]" = torch.ops.prims.fma.default(neg_19, sum_282, mul_505);  neg_19 = sum_282 = mul_505 = None
        convert_element_type_500: "bf16[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(fma_19, torch.bfloat16)
        sum_283: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_19, [0], True, dtype = torch.float32);  fma_19 = None
        squeeze_19: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_283, 0);  sum_283 = None
        permute_217: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_19, [1, 2, 0]);  squeeze_19 = None
        view_691: "f32[2401, 16]" = torch.ops.aten.view.default(permute_217, _shape_param_674);  permute_217 = _shape_param_674 = None
        index_put_19: "f32[169, 16]" = torch.ops.aten.index_put.default(full_1, [view_689], view_691, True);  full_1 = view_689 = view_691 = None
        view_692: "bf16[8192, 49, 49]" = torch.ops.aten.view.default(convert_element_type_500, _shape_param_675);  convert_element_type_500 = _shape_param_675 = None
        bmm_78: "bf16[8192, 32, 49]" = torch.ops.aten.bmm.default(arg574_1, view_692);  arg574_1 = None
        bmm_79: "bf16[8192, 49, 32]" = torch.ops.aten.bmm.default(view_692, arg575_1);  view_692 = arg575_1 = None
        view_693: "bf16[512, 16, 32, 49]" = torch.ops.aten.view.default(bmm_78, _shape_param_676);  bmm_78 = _shape_param_676 = None
        view_694: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(bmm_79, _shape_param_677);  bmm_79 = _shape_param_677 = None
        permute_218: "bf16[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_693, [0, 1, 3, 2]);  view_693 = None
        mul_506: "bf16[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_694, 0.1767766952966369);  view_694 = None
        cat_19: "bf16[1536, 16, 49, 32]" = torch.ops.aten.cat.default([mul_506, permute_218, view_686]);  mul_506 = permute_218 = view_686 = None
        view_695: "bf16[3, 512, 16, 49, 32]" = torch.ops.aten.view.default(cat_19, _shape_param_678);  cat_19 = _shape_param_678 = None
        permute_219: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.permute.default(view_695, [1, 3, 0, 2, 4]);  view_695 = None
        clone_95: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_219, memory_format = torch.contiguous_format);  permute_219 = None
        view_696: "bf16[512, 49, 1536]" = torch.ops.aten.view.default(clone_95, _shape_param_679);  clone_95 = _shape_param_679 = None
        view_697: "bf16[25088, 1536]" = torch.ops.aten.view.default(view_696, _shape_param_680);  view_696 = _shape_param_680 = None
        mm_161: "bf16[25088, 512]" = torch.ops.aten.mm.default(view_697, arg576_1);  arg576_1 = None
        permute_220: "bf16[1536, 25088]" = torch.ops.aten.permute.default(view_697, [1, 0])
        mm_162: "bf16[1536, 512]" = torch.ops.aten.mm.default(permute_220, arg577_1);  permute_220 = arg577_1 = None
        sum_284: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_697, [0], True, dtype = torch.float32);  view_697 = None
        view_698: "f32[1536]" = torch.ops.aten.view.default(sum_284, _shape_param_681);  sum_284 = _shape_param_681 = None
        convert_element_type_501: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_698, torch.bfloat16);  view_698 = None
        view_699: "bf16[512, 49, 512]" = torch.ops.aten.view.default(mm_161, _shape_param_682);  mm_161 = _shape_param_682 = None
        convert_element_type_502: "f32[512, 49, 512]" = torch.ops.prims.convert_element_type.default(view_699, torch.float32);  view_699 = None
        convert_element_type_503: "f32[1536, 512]" = torch.ops.prims.convert_element_type.default(mm_162, torch.float32);  mm_162 = None
        convert_element_type_504: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_501, torch.float32);  convert_element_type_501 = None
        view_700: "f32[512, 7, 7, 512]" = torch.ops.aten.view.default(convert_element_type_502, _shape_param_683);  convert_element_type_502 = _shape_param_683 = None
        view_701: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_700, _shape_param_684);  view_700 = _shape_param_684 = None
        permute_221: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_701, [0, 1, 3, 2, 4, 5]);  view_701 = None
        clone_96: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_221, memory_format = torch.contiguous_format);  permute_221 = None
        view_702: "f32[128, 14, 14, 512]" = torch.ops.aten.view.default(clone_96, _shape_param_685);  clone_96 = _shape_param_685 = None
        mul_507: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_702, arg578_1);  arg578_1 = None
        mul_508: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_507, 512)
        sum_285: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_507, [3], True)
        view_703: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(arg579_1, _shape_param_686);  arg579_1 = _shape_param_686 = None
        convert_element_type_505: "f32[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(view_703, torch.float32);  view_703 = None
        sub_143: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_505, arg580_1);  convert_element_type_505 = arg580_1 = None
        mul_509: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_143, arg581_1);  sub_143 = None
        mul_510: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_507, mul_509);  mul_507 = None
        sum_286: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_510, [3], True);  mul_510 = None
        mul_511: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_509, sum_286);  sum_286 = None
        sub_144: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_508, sum_285);  mul_508 = sum_285 = None
        sub_145: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_144, mul_511);  sub_144 = mul_511 = None
        div_102: "f32[128, 14, 14, 1]" = torch.ops.aten.div.Tensor(arg581_1, 512);  arg581_1 = None
        mul_512: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_102, sub_145);  div_102 = sub_145 = None
        mul_513: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_702, mul_509);  mul_509 = None
        sum_287: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_513, [0, 1, 2]);  mul_513 = None
        sum_288: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_702, [0, 1, 2]);  view_702 = None
        convert_element_type_506: "bf16[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(mul_512, torch.bfloat16);  mul_512 = None
        add_109: "bf16[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_677, convert_element_type_506);  view_677 = convert_element_type_506 = None
        view_704: "bf16[25088, 512]" = torch.ops.aten.view.default(add_109, _shape_param_687);  add_109 = _shape_param_687 = None
        permute_222: "bf16[512, 25088]" = torch.ops.aten.permute.default(view_704, [1, 0])
        mm_163: "bf16[512, 1024]" = torch.ops.aten.mm.default(permute_222, arg582_1);  permute_222 = arg582_1 = None
        mm_164: "bf16[25088, 1024]" = torch.ops.aten.mm.default(view_704, arg583_1);  view_704 = arg583_1 = None
        view_705: "bf16[128, 14, 14, 1024]" = torch.ops.aten.view.default(mm_164, _shape_param_688);  mm_164 = _shape_param_688 = None
        convert_element_type_507: "f32[128, 14, 14, 1024]" = torch.ops.prims.convert_element_type.default(view_705, torch.float32);  view_705 = None
        convert_element_type_508: "f32[512, 1024]" = torch.ops.prims.convert_element_type.default(mm_163, torch.float32);  mm_163 = None
        mul_514: "f32[128, 14, 14, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_507, arg584_1);  arg584_1 = None
        mul_515: "f32[128, 14, 14, 1024]" = torch.ops.aten.mul.Tensor(mul_514, 1024)
        sum_289: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_514, [3], True)
        convert_element_type_509: "f32[128, 14, 14, 1024]" = torch.ops.prims.convert_element_type.default(arg585_1, torch.float32);  arg585_1 = None
        sub_146: "f32[128, 14, 14, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_509, arg586_1);  convert_element_type_509 = arg586_1 = None
        mul_516: "f32[128, 14, 14, 1024]" = torch.ops.aten.mul.Tensor(sub_146, arg587_1);  sub_146 = None
        mul_517: "f32[128, 14, 14, 1024]" = torch.ops.aten.mul.Tensor(mul_514, mul_516);  mul_514 = None
        sum_290: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_517, [3], True);  mul_517 = None
        mul_518: "f32[128, 14, 14, 1024]" = torch.ops.aten.mul.Tensor(mul_516, sum_290);  sum_290 = None
        sub_147: "f32[128, 14, 14, 1024]" = torch.ops.aten.sub.Tensor(mul_515, sum_289);  mul_515 = sum_289 = None
        sub_148: "f32[128, 14, 14, 1024]" = torch.ops.aten.sub.Tensor(sub_147, mul_518);  sub_147 = mul_518 = None
        div_103: "f32[128, 14, 14, 1]" = torch.ops.aten.div.Tensor(arg587_1, 1024);  arg587_1 = None
        mul_519: "f32[128, 14, 14, 1024]" = torch.ops.aten.mul.Tensor(div_103, sub_148);  div_103 = sub_148 = None
        mul_520: "f32[128, 14, 14, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_507, mul_516);  mul_516 = None
        sum_291: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_520, [0, 1, 2]);  mul_520 = None
        sum_292: "f32[1024]" = torch.ops.aten.sum.dim_IntList(convert_element_type_507, [0, 1, 2]);  convert_element_type_507 = None
        convert_element_type_510: "bf16[128, 14, 14, 1024]" = torch.ops.prims.convert_element_type.default(mul_519, torch.bfloat16);  mul_519 = None
        view_706: "bf16[128, 14, 14, 2, 2, 256]" = torch.ops.aten.view.default(convert_element_type_510, _shape_param_689);  convert_element_type_510 = _shape_param_689 = None
        permute_223: "bf16[128, 14, 2, 14, 2, 256]" = torch.ops.aten.permute.default(view_706, [0, 1, 4, 2, 3, 5]);  view_706 = None
        clone_97: "bf16[128, 14, 2, 14, 2, 256]" = torch.ops.aten.clone.default(permute_223, memory_format = torch.contiguous_format);  permute_223 = None
        view_707: "bf16[128, 28, 28, 256]" = torch.ops.aten.view.default(clone_97, _shape_param_690);  clone_97 = _shape_param_690 = None
        view_708: "bf16[128, 784, 256]" = torch.ops.aten.view.default(view_707, _shape_param_691);  view_707 = _shape_param_691 = None
        convert_element_type_511: "bf16[128, 1, 1]" = torch.ops.prims.convert_element_type.default(arg588_1, torch.bfloat16);  arg588_1 = None
        div_104: "bf16[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_511, 0.9869565209373832);  convert_element_type_511 = None
        mul_521: "bf16[128, 784, 256]" = torch.ops.aten.mul.Tensor(view_708, div_104);  div_104 = None
        view_709: "bf16[100352, 256]" = torch.ops.aten.view.default(mul_521, _shape_param_692);  mul_521 = _shape_param_692 = None
        mm_165: "bf16[100352, 1024]" = torch.ops.aten.mm.default(view_709, arg589_1);  arg589_1 = None
        permute_224: "bf16[256, 100352]" = torch.ops.aten.permute.default(view_709, [1, 0])
        mm_166: "bf16[256, 1024]" = torch.ops.aten.mm.default(permute_224, arg590_1);  permute_224 = arg590_1 = None
        sum_293: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_709, [0], True, dtype = torch.float32);  view_709 = None
        view_710: "f32[256]" = torch.ops.aten.view.default(sum_293, _shape_param_693);  sum_293 = _shape_param_693 = None
        convert_element_type_512: "bf16[256]" = torch.ops.prims.convert_element_type.default(view_710, torch.bfloat16);  view_710 = None
        view_711: "bf16[128, 784, 1024]" = torch.ops.aten.view.default(mm_165, _shape_param_694);  mm_165 = _shape_param_694 = None
        convert_element_type_513: "f32[256, 1024]" = torch.ops.prims.convert_element_type.default(mm_166, torch.float32);  mm_166 = None
        convert_element_type_514: "f32[256]" = torch.ops.prims.convert_element_type.default(convert_element_type_512, torch.float32);  convert_element_type_512 = None
        convert_element_type_515: "f32[128, 784, 1024]" = torch.ops.prims.convert_element_type.default(view_711, torch.float32);  view_711 = None
        view_712: "bf16[128, 784, 1024]" = torch.ops.aten.view.default(arg591_1, _shape_param_695);  arg591_1 = _shape_param_695 = None
        convert_element_type_516: "f32[128, 784, 1024]" = torch.ops.prims.convert_element_type.default(view_712, torch.float32);  view_712 = None
        mul_522: "f32[128, 784, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_516, 0.7071067811865476)
        erf_20: "f32[128, 784, 1024]" = torch.ops.aten.erf.default(mul_522);  mul_522 = None
        add_110: "f32[128, 784, 1024]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
        mul_523: "f32[128, 784, 1024]" = torch.ops.aten.mul.Tensor(add_110, 0.5);  add_110 = None
        mul_524: "f32[128, 784, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_516, convert_element_type_516)
        mul_525: "f32[128, 784, 1024]" = torch.ops.aten.mul.Tensor(mul_524, -0.5);  mul_524 = None
        exp_40: "f32[128, 784, 1024]" = torch.ops.aten.exp.default(mul_525);  mul_525 = None
        mul_526: "f32[128, 784, 1024]" = torch.ops.aten.mul.Tensor(exp_40, 0.3989422804014327);  exp_40 = None
        mul_527: "f32[128, 784, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_516, mul_526);  convert_element_type_516 = mul_526 = None
        add_111: "f32[128, 784, 1024]" = torch.ops.aten.add.Tensor(mul_523, mul_527);  mul_523 = mul_527 = None
        mul_528: "f32[128, 784, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_515, add_111);  convert_element_type_515 = add_111 = None
        convert_element_type_517: "bf16[128, 784, 1024]" = torch.ops.prims.convert_element_type.default(mul_528, torch.bfloat16);  mul_528 = None
        view_713: "bf16[100352, 1024]" = torch.ops.aten.view.default(convert_element_type_517, _shape_param_696);  convert_element_type_517 = _shape_param_696 = None
        mm_167: "bf16[100352, 256]" = torch.ops.aten.mm.default(view_713, arg592_1);  arg592_1 = None
        permute_225: "bf16[1024, 100352]" = torch.ops.aten.permute.default(view_713, [1, 0])
        mm_168: "bf16[1024, 256]" = torch.ops.aten.mm.default(permute_225, arg593_1);  permute_225 = arg593_1 = None
        sum_294: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_713, [0], True, dtype = torch.float32);  view_713 = None
        view_714: "f32[1024]" = torch.ops.aten.view.default(sum_294, _shape_param_697);  sum_294 = _shape_param_697 = None
        convert_element_type_518: "bf16[1024]" = torch.ops.prims.convert_element_type.default(view_714, torch.bfloat16);  view_714 = None
        view_715: "bf16[128, 784, 256]" = torch.ops.aten.view.default(mm_167, _shape_param_698);  mm_167 = _shape_param_698 = None
        convert_element_type_519: "f32[128, 784, 256]" = torch.ops.prims.convert_element_type.default(view_715, torch.float32);  view_715 = None
        convert_element_type_520: "f32[1024, 256]" = torch.ops.prims.convert_element_type.default(mm_168, torch.float32);  mm_168 = None
        convert_element_type_521: "f32[1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_518, torch.float32);  convert_element_type_518 = None
        mul_529: "f32[128, 784, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_519, arg594_1);  arg594_1 = None
        mul_530: "f32[128, 784, 256]" = torch.ops.aten.mul.Tensor(mul_529, 256)
        sum_295: "f32[128, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_529, [2], True)
        convert_element_type_522: "f32[128, 784, 256]" = torch.ops.prims.convert_element_type.default(arg595_1, torch.float32);  arg595_1 = None
        sub_149: "f32[128, 784, 256]" = torch.ops.aten.sub.Tensor(convert_element_type_522, arg596_1);  convert_element_type_522 = arg596_1 = None
        mul_531: "f32[128, 784, 256]" = torch.ops.aten.mul.Tensor(sub_149, arg597_1);  sub_149 = None
        mul_532: "f32[128, 784, 256]" = torch.ops.aten.mul.Tensor(mul_529, mul_531);  mul_529 = None
        sum_296: "f32[128, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_532, [2], True);  mul_532 = None
        mul_533: "f32[128, 784, 256]" = torch.ops.aten.mul.Tensor(mul_531, sum_296);  sum_296 = None
        sub_150: "f32[128, 784, 256]" = torch.ops.aten.sub.Tensor(mul_530, sum_295);  mul_530 = sum_295 = None
        sub_151: "f32[128, 784, 256]" = torch.ops.aten.sub.Tensor(sub_150, mul_533);  sub_150 = mul_533 = None
        div_105: "f32[128, 784, 1]" = torch.ops.aten.div.Tensor(arg597_1, 256);  arg597_1 = None
        mul_534: "f32[128, 784, 256]" = torch.ops.aten.mul.Tensor(div_105, sub_151);  div_105 = sub_151 = None
        mul_535: "f32[128, 784, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_519, mul_531);  mul_531 = None
        sum_297: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_535, [0, 1]);  mul_535 = None
        sum_298: "f32[256]" = torch.ops.aten.sum.dim_IntList(convert_element_type_519, [0, 1]);  convert_element_type_519 = None
        convert_element_type_523: "bf16[128, 784, 256]" = torch.ops.prims.convert_element_type.default(mul_534, torch.bfloat16);  mul_534 = None
        add_112: "bf16[128, 784, 256]" = torch.ops.aten.add.Tensor(view_708, convert_element_type_523);  view_708 = convert_element_type_523 = None
        view_716: "bf16[128, 28, 28, 256]" = torch.ops.aten.view.default(add_112, _shape_param_699);  add_112 = _shape_param_699 = None
        convert_element_type_524: "bf16[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(arg598_1, torch.bfloat16);  arg598_1 = None
        div_106: "bf16[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_524, 0.9869565209373832);  convert_element_type_524 = None
        mul_536: "bf16[128, 28, 28, 256]" = torch.ops.aten.mul.Tensor(view_716, div_106);  div_106 = None
        iota_1: "i64[28]" = torch.ops.prims.iota.default(28, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_113: "i64[28]" = torch.ops.aten.add.Tensor(iota_1, 3);  iota_1 = None
        fmod_1: "i64[28]" = torch.ops.aten.fmod.Scalar(add_113, 28);  add_113 = None
        index_56: "bf16[128, 28, 28, 256]" = torch.ops.aten.index.Tensor(mul_536, [None, None, fmod_1]);  mul_536 = None
        index_57: "bf16[128, 28, 28, 256]" = torch.ops.aten.index.Tensor(index_56, [None, fmod_1]);  index_56 = fmod_1 = None
        view_717: "bf16[128, 4, 7, 4, 7, 256]" = torch.ops.aten.view.default(index_57, _shape_param_700);  index_57 = _shape_param_700 = None
        permute_226: "bf16[128, 4, 4, 7, 7, 256]" = torch.ops.aten.permute.default(view_717, [0, 1, 3, 2, 4, 5]);  view_717 = None
        clone_98: "bf16[128, 4, 4, 7, 7, 256]" = torch.ops.aten.clone.default(permute_226, memory_format = torch.contiguous_format);  permute_226 = None
        view_718: "bf16[2048, 7, 7, 256]" = torch.ops.aten.view.default(clone_98, _shape_param_701);  clone_98 = _shape_param_701 = None
        view_719: "bf16[2048, 49, 256]" = torch.ops.aten.view.default(view_718, _shape_param_702);  view_718 = _shape_param_702 = None
        view_720: "bf16[100352, 256]" = torch.ops.aten.view.default(view_719, _shape_param_703);  view_719 = _shape_param_703 = None
        mm_169: "bf16[100352, 256]" = torch.ops.aten.mm.default(view_720, arg599_1);  arg599_1 = None
        permute_227: "bf16[256, 100352]" = torch.ops.aten.permute.default(view_720, [1, 0])
        mm_170: "bf16[256, 256]" = torch.ops.aten.mm.default(permute_227, arg600_1);  permute_227 = arg600_1 = None
        sum_299: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_720, [0], True, dtype = torch.float32);  view_720 = None
        view_721: "f32[256]" = torch.ops.aten.view.default(sum_299, _shape_param_704);  sum_299 = _shape_param_704 = None
        convert_element_type_525: "bf16[256]" = torch.ops.prims.convert_element_type.default(view_721, torch.bfloat16);  view_721 = None
        view_722: "bf16[2048, 49, 256]" = torch.ops.aten.view.default(mm_169, _shape_param_705);  mm_169 = _shape_param_705 = None
        convert_element_type_526: "f32[256, 256]" = torch.ops.prims.convert_element_type.default(mm_170, torch.float32);  mm_170 = None
        convert_element_type_527: "f32[256]" = torch.ops.prims.convert_element_type.default(convert_element_type_525, torch.float32);  convert_element_type_525 = None
        view_723: "bf16[2048, 49, 8, 32]" = torch.ops.aten.view.default(view_722, _shape_param_706);  view_722 = _shape_param_706 = None
        permute_228: "bf16[2048, 8, 49, 32]" = torch.ops.aten.permute.default(view_723, [0, 2, 1, 3]);  view_723 = None
        clone_99: "bf16[2048, 8, 49, 32]" = torch.ops.aten.clone.default(permute_228, memory_format = torch.contiguous_format);  permute_228 = None
        view_724: "bf16[16384, 49, 32]" = torch.ops.aten.view.default(clone_99, _shape_param_707);  clone_99 = _shape_param_707 = None
        bmm_80: "bf16[16384, 49, 32]" = torch.ops.aten.bmm.default(arg601_1, view_724);  arg601_1 = None
        bmm_81: "bf16[16384, 49, 49]" = torch.ops.aten.bmm.default(view_724, arg602_1);  view_724 = arg602_1 = None
        view_725: "bf16[2048, 8, 49, 32]" = torch.ops.aten.view.default(bmm_80, _shape_param_708);  bmm_80 = _shape_param_708 = None
        view_726: "bf16[2048, 8, 49, 49]" = torch.ops.aten.view.default(bmm_81, _shape_param_709);  bmm_81 = _shape_param_709 = None
        convert_element_type_528: "f32[2048, 8, 49, 49]" = torch.ops.prims.convert_element_type.default(view_726, torch.float32);  view_726 = None
        slice_37: "bf16[16384, 49, 56]" = torch.ops.aten.slice.Tensor(arg603_1, 1, 0, -7);  arg603_1 = None
        slice_38: "bf16[16384, 49, 49]" = torch.ops.aten.slice.Tensor(slice_37, 2, 0, -7);  slice_37 = None
        view_727: "bf16[2048, 8, 49, 49]" = torch.ops.aten.view.default(slice_38, _shape_param_710);  slice_38 = _shape_param_710 = None
        view_728: "i64[2401]" = torch.ops.aten.view.default(arg604_1, [-1]);  arg604_1 = None
        index_58: "f32[2401, 8]" = torch.ops.aten.index.Tensor(arg605_1, [view_728]);  arg605_1 = None
        view_729: "f32[49, 49, 8]" = torch.ops.aten.view.default(index_58, _shape_param_711);  index_58 = _shape_param_711 = None
        permute_229: "f32[8, 49, 49]" = torch.ops.aten.permute.default(view_729, [2, 0, 1]);  view_729 = None
        clone_100: "f32[8, 49, 49]" = torch.ops.aten.clone.default(permute_229, memory_format = torch.contiguous_format);  permute_229 = None
        unsqueeze_40: "f32[1, 8, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_100, 0);  clone_100 = None
        add_114: "f32[2048, 8, 49, 49]" = torch.ops.aten.add.Tensor(view_727, unsqueeze_40);  view_727 = unsqueeze_40 = None
        view_730: "f32[128, 16, 8, 49, 49]" = torch.ops.aten.view.default(add_114, _shape_param_712);  add_114 = _shape_param_712 = None
        unsqueeze_41: "f32[16, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(arg606_1, 1);  arg606_1 = None
        unsqueeze_42: "f32[1, 16, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_41, 0);  unsqueeze_41 = None
        add_115: "f32[128, 16, 8, 49, 49]" = torch.ops.aten.add.Tensor(view_730, unsqueeze_42);  view_730 = unsqueeze_42 = None
        view_731: "f32[2048, 8, 49, 49]" = torch.ops.aten.view.default(add_115, _shape_param_713);  add_115 = _shape_param_713 = None
        sub_152: "f32[2048, 8, 49, 49]" = torch.ops.aten.sub.Tensor(view_731, arg607_1);  view_731 = arg607_1 = None
        exp_41: "f32[2048, 8, 49, 49]" = torch.ops.aten.exp.default(sub_152);  sub_152 = None
        div_107: "f32[2048, 8, 49, 49]" = torch.ops.aten.div.Tensor(exp_41, arg608_1);  exp_41 = arg608_1 = None
        mul_537: "f32[2048, 8, 49, 49]" = torch.ops.aten.mul.Tensor(convert_element_type_528, div_107);  convert_element_type_528 = None
        sum_300: "f32[2048, 8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_537, [-1], True)
        neg_20: "f32[2048, 8, 49, 49]" = torch.ops.aten.neg.default(div_107);  div_107 = None
        fma_20: "f32[2048, 8, 49, 49]" = torch.ops.prims.fma.default(neg_20, sum_300, mul_537);  neg_20 = sum_300 = mul_537 = None
        convert_element_type_529: "bf16[2048, 8, 49, 49]" = torch.ops.prims.convert_element_type.default(fma_20, torch.bfloat16)
        sum_301: "f32[1, 8, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_20, [0], True, dtype = torch.float32);  fma_20 = None
        squeeze_20: "f32[8, 49, 49]" = torch.ops.aten.squeeze.dim(sum_301, 0);  sum_301 = None
        permute_230: "f32[49, 49, 8]" = torch.ops.aten.permute.default(squeeze_20, [1, 2, 0]);  squeeze_20 = None
        view_732: "f32[2401, 8]" = torch.ops.aten.view.default(permute_230, _shape_param_714);  permute_230 = _shape_param_714 = None
        full_2: "f32[169, 8]" = torch.ops.aten.full.default(_shape_param_715, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_715 = None
        index_put_20: "f32[169, 8]" = torch.ops.aten.index_put.default(full_2, [view_728], view_732, True);  view_728 = view_732 = None
        view_733: "bf16[16384, 49, 49]" = torch.ops.aten.view.default(convert_element_type_529, _shape_param_716);  convert_element_type_529 = _shape_param_716 = None
        bmm_82: "bf16[16384, 32, 49]" = torch.ops.aten.bmm.default(arg609_1, view_733);  arg609_1 = None
        bmm_83: "bf16[16384, 49, 32]" = torch.ops.aten.bmm.default(view_733, arg610_1);  view_733 = arg610_1 = None
        view_734: "bf16[2048, 8, 32, 49]" = torch.ops.aten.view.default(bmm_82, _shape_param_717);  bmm_82 = _shape_param_717 = None
        view_735: "bf16[2048, 8, 49, 32]" = torch.ops.aten.view.default(bmm_83, _shape_param_718);  bmm_83 = _shape_param_718 = None
        permute_231: "bf16[2048, 8, 49, 32]" = torch.ops.aten.permute.default(view_734, [0, 1, 3, 2]);  view_734 = None
        mul_538: "bf16[2048, 8, 49, 32]" = torch.ops.aten.mul.Tensor(view_735, 0.1767766952966369);  view_735 = None
        cat_20: "bf16[6144, 8, 49, 32]" = torch.ops.aten.cat.default([mul_538, permute_231, view_725]);  mul_538 = permute_231 = view_725 = None
        view_736: "bf16[3, 2048, 8, 49, 32]" = torch.ops.aten.view.default(cat_20, _shape_param_719);  cat_20 = _shape_param_719 = None
        permute_232: "bf16[2048, 49, 3, 8, 32]" = torch.ops.aten.permute.default(view_736, [1, 3, 0, 2, 4]);  view_736 = None
        clone_101: "bf16[2048, 49, 3, 8, 32]" = torch.ops.aten.clone.default(permute_232, memory_format = torch.contiguous_format);  permute_232 = None
        view_737: "bf16[2048, 49, 768]" = torch.ops.aten.view.default(clone_101, _shape_param_720);  clone_101 = _shape_param_720 = None
        view_738: "bf16[100352, 768]" = torch.ops.aten.view.default(view_737, _shape_param_721);  view_737 = _shape_param_721 = None
        mm_171: "bf16[100352, 256]" = torch.ops.aten.mm.default(view_738, arg611_1);  arg611_1 = None
        permute_233: "bf16[768, 100352]" = torch.ops.aten.permute.default(view_738, [1, 0])
        mm_172: "bf16[768, 256]" = torch.ops.aten.mm.default(permute_233, arg612_1);  permute_233 = arg612_1 = None
        sum_302: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_738, [0], True, dtype = torch.float32);  view_738 = None
        view_739: "f32[768]" = torch.ops.aten.view.default(sum_302, _shape_param_722);  sum_302 = _shape_param_722 = None
        convert_element_type_530: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_739, torch.bfloat16);  view_739 = None
        view_740: "bf16[2048, 49, 256]" = torch.ops.aten.view.default(mm_171, _shape_param_723);  mm_171 = _shape_param_723 = None
        convert_element_type_531: "f32[2048, 49, 256]" = torch.ops.prims.convert_element_type.default(view_740, torch.float32);  view_740 = None
        convert_element_type_532: "f32[768, 256]" = torch.ops.prims.convert_element_type.default(mm_172, torch.float32);  mm_172 = None
        convert_element_type_533: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_530, torch.float32);  convert_element_type_530 = None
        view_741: "f32[2048, 7, 7, 256]" = torch.ops.aten.view.default(convert_element_type_531, _shape_param_724);  convert_element_type_531 = _shape_param_724 = None
        view_742: "f32[128, 4, 4, 7, 7, 256]" = torch.ops.aten.view.default(view_741, _shape_param_725);  view_741 = _shape_param_725 = None
        permute_234: "f32[128, 4, 7, 4, 7, 256]" = torch.ops.aten.permute.default(view_742, [0, 1, 3, 2, 4, 5]);  view_742 = None
        clone_102: "f32[128, 4, 7, 4, 7, 256]" = torch.ops.aten.clone.default(permute_234, memory_format = torch.contiguous_format);  permute_234 = None
        view_743: "f32[128, 28, 28, 256]" = torch.ops.aten.view.default(clone_102, _shape_param_726);  clone_102 = _shape_param_726 = None
        index_59: "f32[128, 28, 28, 256]" = torch.ops.aten.index.Tensor(view_743, [None, None, arg613_1]);  view_743 = None
        index_60: "f32[128, 28, 28, 256]" = torch.ops.aten.index.Tensor(index_59, [None, arg613_1]);  index_59 = arg613_1 = None
        mul_539: "f32[128, 28, 28, 256]" = torch.ops.aten.mul.Tensor(index_60, arg614_1);  arg614_1 = None
        mul_540: "f32[128, 28, 28, 256]" = torch.ops.aten.mul.Tensor(mul_539, 256)
        sum_303: "f32[128, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_539, [3], True)
        convert_element_type_534: "f32[128, 28, 28, 256]" = torch.ops.prims.convert_element_type.default(arg615_1, torch.float32);  arg615_1 = None
        sub_153: "f32[128, 28, 28, 256]" = torch.ops.aten.sub.Tensor(convert_element_type_534, arg616_1);  convert_element_type_534 = arg616_1 = None
        mul_541: "f32[128, 28, 28, 256]" = torch.ops.aten.mul.Tensor(sub_153, arg617_1);  sub_153 = None
        mul_542: "f32[128, 28, 28, 256]" = torch.ops.aten.mul.Tensor(mul_539, mul_541);  mul_539 = None
        sum_304: "f32[128, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_542, [3], True);  mul_542 = None
        mul_543: "f32[128, 28, 28, 256]" = torch.ops.aten.mul.Tensor(mul_541, sum_304);  sum_304 = None
        sub_154: "f32[128, 28, 28, 256]" = torch.ops.aten.sub.Tensor(mul_540, sum_303);  mul_540 = sum_303 = None
        sub_155: "f32[128, 28, 28, 256]" = torch.ops.aten.sub.Tensor(sub_154, mul_543);  sub_154 = mul_543 = None
        div_108: "f32[128, 28, 28, 1]" = torch.ops.aten.div.Tensor(arg617_1, 256);  arg617_1 = None
        mul_544: "f32[128, 28, 28, 256]" = torch.ops.aten.mul.Tensor(div_108, sub_155);  div_108 = sub_155 = None
        mul_545: "f32[128, 28, 28, 256]" = torch.ops.aten.mul.Tensor(index_60, mul_541);  mul_541 = None
        sum_305: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_545, [0, 1, 2]);  mul_545 = None
        sum_306: "f32[256]" = torch.ops.aten.sum.dim_IntList(index_60, [0, 1, 2]);  index_60 = None
        convert_element_type_535: "bf16[128, 28, 28, 256]" = torch.ops.prims.convert_element_type.default(mul_544, torch.bfloat16);  mul_544 = None
        add_116: "bf16[128, 28, 28, 256]" = torch.ops.aten.add.Tensor(view_716, convert_element_type_535);  view_716 = convert_element_type_535 = None
        view_744: "bf16[128, 784, 256]" = torch.ops.aten.view.default(add_116, _shape_param_727);  add_116 = _shape_param_727 = None
        convert_element_type_536: "bf16[128, 1, 1]" = torch.ops.prims.convert_element_type.default(arg618_1, torch.bfloat16);  arg618_1 = None
        div_109: "bf16[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_536, 0.9913043472915888);  convert_element_type_536 = None
        mul_546: "bf16[128, 784, 256]" = torch.ops.aten.mul.Tensor(view_744, div_109);  div_109 = None
        view_745: "bf16[100352, 256]" = torch.ops.aten.view.default(mul_546, _shape_param_728);  mul_546 = _shape_param_728 = None
        mm_173: "bf16[100352, 1024]" = torch.ops.aten.mm.default(view_745, arg619_1);  arg619_1 = None
        permute_235: "bf16[256, 100352]" = torch.ops.aten.permute.default(view_745, [1, 0])
        mm_174: "bf16[256, 1024]" = torch.ops.aten.mm.default(permute_235, arg620_1);  permute_235 = arg620_1 = None
        sum_307: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_745, [0], True, dtype = torch.float32);  view_745 = None
        view_746: "f32[256]" = torch.ops.aten.view.default(sum_307, _shape_param_729);  sum_307 = _shape_param_729 = None
        convert_element_type_537: "bf16[256]" = torch.ops.prims.convert_element_type.default(view_746, torch.bfloat16);  view_746 = None
        view_747: "bf16[128, 784, 1024]" = torch.ops.aten.view.default(mm_173, _shape_param_730);  mm_173 = _shape_param_730 = None
        convert_element_type_538: "f32[256, 1024]" = torch.ops.prims.convert_element_type.default(mm_174, torch.float32);  mm_174 = None
        convert_element_type_539: "f32[256]" = torch.ops.prims.convert_element_type.default(convert_element_type_537, torch.float32);  convert_element_type_537 = None
        convert_element_type_540: "f32[128, 784, 1024]" = torch.ops.prims.convert_element_type.default(view_747, torch.float32);  view_747 = None
        view_748: "bf16[128, 784, 1024]" = torch.ops.aten.view.default(arg621_1, _shape_param_731);  arg621_1 = _shape_param_731 = None
        convert_element_type_541: "f32[128, 784, 1024]" = torch.ops.prims.convert_element_type.default(view_748, torch.float32);  view_748 = None
        mul_547: "f32[128, 784, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_541, 0.7071067811865476)
        erf_21: "f32[128, 784, 1024]" = torch.ops.aten.erf.default(mul_547);  mul_547 = None
        add_117: "f32[128, 784, 1024]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
        mul_548: "f32[128, 784, 1024]" = torch.ops.aten.mul.Tensor(add_117, 0.5);  add_117 = None
        mul_549: "f32[128, 784, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_541, convert_element_type_541)
        mul_550: "f32[128, 784, 1024]" = torch.ops.aten.mul.Tensor(mul_549, -0.5);  mul_549 = None
        exp_42: "f32[128, 784, 1024]" = torch.ops.aten.exp.default(mul_550);  mul_550 = None
        mul_551: "f32[128, 784, 1024]" = torch.ops.aten.mul.Tensor(exp_42, 0.3989422804014327);  exp_42 = None
        mul_552: "f32[128, 784, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_541, mul_551);  convert_element_type_541 = mul_551 = None
        add_118: "f32[128, 784, 1024]" = torch.ops.aten.add.Tensor(mul_548, mul_552);  mul_548 = mul_552 = None
        mul_553: "f32[128, 784, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_540, add_118);  convert_element_type_540 = add_118 = None
        convert_element_type_542: "bf16[128, 784, 1024]" = torch.ops.prims.convert_element_type.default(mul_553, torch.bfloat16);  mul_553 = None
        view_749: "bf16[100352, 1024]" = torch.ops.aten.view.default(convert_element_type_542, _shape_param_732);  convert_element_type_542 = _shape_param_732 = None
        mm_175: "bf16[100352, 256]" = torch.ops.aten.mm.default(view_749, arg622_1);  arg622_1 = None
        permute_236: "bf16[1024, 100352]" = torch.ops.aten.permute.default(view_749, [1, 0])
        mm_176: "bf16[1024, 256]" = torch.ops.aten.mm.default(permute_236, arg623_1);  permute_236 = arg623_1 = None
        sum_308: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_749, [0], True, dtype = torch.float32);  view_749 = None
        view_750: "f32[1024]" = torch.ops.aten.view.default(sum_308, _shape_param_733);  sum_308 = _shape_param_733 = None
        convert_element_type_543: "bf16[1024]" = torch.ops.prims.convert_element_type.default(view_750, torch.bfloat16);  view_750 = None
        view_751: "bf16[128, 784, 256]" = torch.ops.aten.view.default(mm_175, _shape_param_734);  mm_175 = _shape_param_734 = None
        convert_element_type_544: "f32[128, 784, 256]" = torch.ops.prims.convert_element_type.default(view_751, torch.float32);  view_751 = None
        convert_element_type_545: "f32[1024, 256]" = torch.ops.prims.convert_element_type.default(mm_176, torch.float32);  mm_176 = None
        convert_element_type_546: "f32[1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_543, torch.float32);  convert_element_type_543 = None
        mul_554: "f32[128, 784, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_544, arg624_1);  arg624_1 = None
        mul_555: "f32[128, 784, 256]" = torch.ops.aten.mul.Tensor(mul_554, 256)
        sum_309: "f32[128, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_554, [2], True)
        convert_element_type_547: "f32[128, 784, 256]" = torch.ops.prims.convert_element_type.default(arg625_1, torch.float32);  arg625_1 = None
        sub_156: "f32[128, 784, 256]" = torch.ops.aten.sub.Tensor(convert_element_type_547, arg626_1);  convert_element_type_547 = arg626_1 = None
        mul_556: "f32[128, 784, 256]" = torch.ops.aten.mul.Tensor(sub_156, arg627_1);  sub_156 = None
        mul_557: "f32[128, 784, 256]" = torch.ops.aten.mul.Tensor(mul_554, mul_556);  mul_554 = None
        sum_310: "f32[128, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_557, [2], True);  mul_557 = None
        mul_558: "f32[128, 784, 256]" = torch.ops.aten.mul.Tensor(mul_556, sum_310);  sum_310 = None
        sub_157: "f32[128, 784, 256]" = torch.ops.aten.sub.Tensor(mul_555, sum_309);  mul_555 = sum_309 = None
        sub_158: "f32[128, 784, 256]" = torch.ops.aten.sub.Tensor(sub_157, mul_558);  sub_157 = mul_558 = None
        div_110: "f32[128, 784, 1]" = torch.ops.aten.div.Tensor(arg627_1, 256);  arg627_1 = None
        mul_559: "f32[128, 784, 256]" = torch.ops.aten.mul.Tensor(div_110, sub_158);  div_110 = sub_158 = None
        mul_560: "f32[128, 784, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_544, mul_556);  mul_556 = None
        sum_311: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_560, [0, 1]);  mul_560 = None
        sum_312: "f32[256]" = torch.ops.aten.sum.dim_IntList(convert_element_type_544, [0, 1]);  convert_element_type_544 = None
        convert_element_type_548: "bf16[128, 784, 256]" = torch.ops.prims.convert_element_type.default(mul_559, torch.bfloat16);  mul_559 = None
        add_119: "bf16[128, 784, 256]" = torch.ops.aten.add.Tensor(view_744, convert_element_type_548);  view_744 = convert_element_type_548 = None
        view_752: "bf16[128, 28, 28, 256]" = torch.ops.aten.view.default(add_119, _shape_param_735);  add_119 = _shape_param_735 = None
        convert_element_type_549: "bf16[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(arg628_1, torch.bfloat16);  arg628_1 = None
        div_111: "bf16[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_549, 0.9913043472915888);  convert_element_type_549 = None
        mul_561: "bf16[128, 28, 28, 256]" = torch.ops.aten.mul.Tensor(view_752, div_111);  div_111 = None
        view_753: "bf16[128, 4, 7, 4, 7, 256]" = torch.ops.aten.view.default(mul_561, _shape_param_736);  mul_561 = _shape_param_736 = None
        permute_237: "bf16[128, 4, 4, 7, 7, 256]" = torch.ops.aten.permute.default(view_753, [0, 1, 3, 2, 4, 5]);  view_753 = None
        clone_103: "bf16[128, 4, 4, 7, 7, 256]" = torch.ops.aten.clone.default(permute_237, memory_format = torch.contiguous_format);  permute_237 = None
        view_754: "bf16[2048, 7, 7, 256]" = torch.ops.aten.view.default(clone_103, _shape_param_737);  clone_103 = _shape_param_737 = None
        view_755: "bf16[2048, 49, 256]" = torch.ops.aten.view.default(view_754, _shape_param_738);  view_754 = _shape_param_738 = None
        view_756: "bf16[100352, 256]" = torch.ops.aten.view.default(view_755, _shape_param_739);  view_755 = _shape_param_739 = None
        mm_177: "bf16[100352, 256]" = torch.ops.aten.mm.default(view_756, arg629_1);  arg629_1 = None
        permute_238: "bf16[256, 100352]" = torch.ops.aten.permute.default(view_756, [1, 0])
        mm_178: "bf16[256, 256]" = torch.ops.aten.mm.default(permute_238, arg630_1);  permute_238 = arg630_1 = None
        sum_313: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_756, [0], True, dtype = torch.float32);  view_756 = None
        view_757: "f32[256]" = torch.ops.aten.view.default(sum_313, _shape_param_740);  sum_313 = _shape_param_740 = None
        convert_element_type_550: "bf16[256]" = torch.ops.prims.convert_element_type.default(view_757, torch.bfloat16);  view_757 = None
        view_758: "bf16[2048, 49, 256]" = torch.ops.aten.view.default(mm_177, _shape_param_741);  mm_177 = _shape_param_741 = None
        convert_element_type_551: "f32[256, 256]" = torch.ops.prims.convert_element_type.default(mm_178, torch.float32);  mm_178 = None
        convert_element_type_552: "f32[256]" = torch.ops.prims.convert_element_type.default(convert_element_type_550, torch.float32);  convert_element_type_550 = None
        view_759: "bf16[2048, 49, 8, 32]" = torch.ops.aten.view.default(view_758, _shape_param_742);  view_758 = _shape_param_742 = None
        permute_239: "bf16[2048, 8, 49, 32]" = torch.ops.aten.permute.default(view_759, [0, 2, 1, 3]);  view_759 = None
        clone_104: "bf16[2048, 8, 49, 32]" = torch.ops.aten.clone.default(permute_239, memory_format = torch.contiguous_format);  permute_239 = None
        view_760: "bf16[16384, 49, 32]" = torch.ops.aten.view.default(clone_104, _shape_param_743);  clone_104 = _shape_param_743 = None
        bmm_84: "bf16[16384, 49, 32]" = torch.ops.aten.bmm.default(arg631_1, view_760);  arg631_1 = None
        bmm_85: "bf16[16384, 49, 49]" = torch.ops.aten.bmm.default(view_760, arg632_1);  view_760 = arg632_1 = None
        view_761: "bf16[2048, 8, 49, 32]" = torch.ops.aten.view.default(bmm_84, _shape_param_744);  bmm_84 = _shape_param_744 = None
        view_762: "bf16[2048, 8, 49, 49]" = torch.ops.aten.view.default(bmm_85, _shape_param_745);  bmm_85 = _shape_param_745 = None
        convert_element_type_553: "f32[2048, 8, 49, 49]" = torch.ops.prims.convert_element_type.default(view_762, torch.float32);  view_762 = None
        slice_39: "bf16[16384, 49, 56]" = torch.ops.aten.slice.Tensor(arg633_1, 1, 0, -7);  arg633_1 = None
        slice_40: "bf16[16384, 49, 49]" = torch.ops.aten.slice.Tensor(slice_39, 2, 0, -7);  slice_39 = None
        view_763: "bf16[2048, 8, 49, 49]" = torch.ops.aten.view.default(slice_40, _shape_param_746);  slice_40 = _shape_param_746 = None
        view_764: "i64[2401]" = torch.ops.aten.view.default(arg634_1, [-1]);  arg634_1 = None
        index_61: "f32[2401, 8]" = torch.ops.aten.index.Tensor(arg635_1, [view_764]);  arg635_1 = None
        view_765: "f32[49, 49, 8]" = torch.ops.aten.view.default(index_61, _shape_param_747);  index_61 = _shape_param_747 = None
        permute_240: "f32[8, 49, 49]" = torch.ops.aten.permute.default(view_765, [2, 0, 1]);  view_765 = None
        clone_105: "f32[8, 49, 49]" = torch.ops.aten.clone.default(permute_240, memory_format = torch.contiguous_format);  permute_240 = None
        unsqueeze_43: "f32[1, 8, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_105, 0);  clone_105 = None
        add_120: "f32[2048, 8, 49, 49]" = torch.ops.aten.add.Tensor(view_763, unsqueeze_43);  view_763 = unsqueeze_43 = None
        sub_159: "f32[2048, 8, 49, 49]" = torch.ops.aten.sub.Tensor(add_120, arg636_1);  add_120 = arg636_1 = None
        exp_43: "f32[2048, 8, 49, 49]" = torch.ops.aten.exp.default(sub_159);  sub_159 = None
        div_112: "f32[2048, 8, 49, 49]" = torch.ops.aten.div.Tensor(exp_43, arg637_1);  exp_43 = arg637_1 = None
        mul_562: "f32[2048, 8, 49, 49]" = torch.ops.aten.mul.Tensor(convert_element_type_553, div_112);  convert_element_type_553 = None
        sum_314: "f32[2048, 8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_562, [-1], True)
        neg_21: "f32[2048, 8, 49, 49]" = torch.ops.aten.neg.default(div_112);  div_112 = None
        fma_21: "f32[2048, 8, 49, 49]" = torch.ops.prims.fma.default(neg_21, sum_314, mul_562);  neg_21 = sum_314 = mul_562 = None
        convert_element_type_554: "bf16[2048, 8, 49, 49]" = torch.ops.prims.convert_element_type.default(fma_21, torch.bfloat16)
        sum_315: "f32[1, 8, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_21, [0], True, dtype = torch.float32);  fma_21 = None
        squeeze_21: "f32[8, 49, 49]" = torch.ops.aten.squeeze.dim(sum_315, 0);  sum_315 = None
        permute_241: "f32[49, 49, 8]" = torch.ops.aten.permute.default(squeeze_21, [1, 2, 0]);  squeeze_21 = None
        view_766: "f32[2401, 8]" = torch.ops.aten.view.default(permute_241, _shape_param_748);  permute_241 = _shape_param_748 = None
        index_put_21: "f32[169, 8]" = torch.ops.aten.index_put.default(full_2, [view_764], view_766, True);  full_2 = view_764 = view_766 = None
        view_767: "bf16[16384, 49, 49]" = torch.ops.aten.view.default(convert_element_type_554, _shape_param_749);  convert_element_type_554 = _shape_param_749 = None
        bmm_86: "bf16[16384, 32, 49]" = torch.ops.aten.bmm.default(arg638_1, view_767);  arg638_1 = None
        bmm_87: "bf16[16384, 49, 32]" = torch.ops.aten.bmm.default(view_767, arg639_1);  view_767 = arg639_1 = None
        view_768: "bf16[2048, 8, 32, 49]" = torch.ops.aten.view.default(bmm_86, _shape_param_750);  bmm_86 = _shape_param_750 = None
        view_769: "bf16[2048, 8, 49, 32]" = torch.ops.aten.view.default(bmm_87, _shape_param_751);  bmm_87 = _shape_param_751 = None
        permute_242: "bf16[2048, 8, 49, 32]" = torch.ops.aten.permute.default(view_768, [0, 1, 3, 2]);  view_768 = None
        mul_563: "bf16[2048, 8, 49, 32]" = torch.ops.aten.mul.Tensor(view_769, 0.1767766952966369);  view_769 = None
        cat_21: "bf16[6144, 8, 49, 32]" = torch.ops.aten.cat.default([mul_563, permute_242, view_761]);  mul_563 = permute_242 = view_761 = None
        view_770: "bf16[3, 2048, 8, 49, 32]" = torch.ops.aten.view.default(cat_21, _shape_param_752);  cat_21 = _shape_param_752 = None
        permute_243: "bf16[2048, 49, 3, 8, 32]" = torch.ops.aten.permute.default(view_770, [1, 3, 0, 2, 4]);  view_770 = None
        clone_106: "bf16[2048, 49, 3, 8, 32]" = torch.ops.aten.clone.default(permute_243, memory_format = torch.contiguous_format);  permute_243 = None
        view_771: "bf16[2048, 49, 768]" = torch.ops.aten.view.default(clone_106, _shape_param_753);  clone_106 = _shape_param_753 = None
        view_772: "bf16[100352, 768]" = torch.ops.aten.view.default(view_771, _shape_param_754);  view_771 = _shape_param_754 = None
        mm_179: "bf16[100352, 256]" = torch.ops.aten.mm.default(view_772, arg640_1);  arg640_1 = None
        permute_244: "bf16[768, 100352]" = torch.ops.aten.permute.default(view_772, [1, 0])
        mm_180: "bf16[768, 256]" = torch.ops.aten.mm.default(permute_244, arg641_1);  permute_244 = arg641_1 = None
        sum_316: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_772, [0], True, dtype = torch.float32);  view_772 = None
        view_773: "f32[768]" = torch.ops.aten.view.default(sum_316, _shape_param_755);  sum_316 = _shape_param_755 = None
        convert_element_type_555: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_773, torch.bfloat16);  view_773 = None
        view_774: "bf16[2048, 49, 256]" = torch.ops.aten.view.default(mm_179, _shape_param_756);  mm_179 = _shape_param_756 = None
        convert_element_type_556: "f32[2048, 49, 256]" = torch.ops.prims.convert_element_type.default(view_774, torch.float32);  view_774 = None
        convert_element_type_557: "f32[768, 256]" = torch.ops.prims.convert_element_type.default(mm_180, torch.float32);  mm_180 = None
        convert_element_type_558: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_555, torch.float32);  convert_element_type_555 = None
        view_775: "f32[2048, 7, 7, 256]" = torch.ops.aten.view.default(convert_element_type_556, _shape_param_757);  convert_element_type_556 = _shape_param_757 = None
        view_776: "f32[128, 4, 4, 7, 7, 256]" = torch.ops.aten.view.default(view_775, _shape_param_758);  view_775 = _shape_param_758 = None
        permute_245: "f32[128, 4, 7, 4, 7, 256]" = torch.ops.aten.permute.default(view_776, [0, 1, 3, 2, 4, 5]);  view_776 = None
        clone_107: "f32[128, 4, 7, 4, 7, 256]" = torch.ops.aten.clone.default(permute_245, memory_format = torch.contiguous_format);  permute_245 = None
        view_777: "f32[128, 28, 28, 256]" = torch.ops.aten.view.default(clone_107, _shape_param_759);  clone_107 = _shape_param_759 = None
        mul_564: "f32[128, 28, 28, 256]" = torch.ops.aten.mul.Tensor(view_777, arg642_1);  arg642_1 = None
        mul_565: "f32[128, 28, 28, 256]" = torch.ops.aten.mul.Tensor(mul_564, 256)
        sum_317: "f32[128, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_564, [3], True)
        view_778: "bf16[128, 28, 28, 256]" = torch.ops.aten.view.default(arg643_1, _shape_param_760);  arg643_1 = _shape_param_760 = None
        convert_element_type_559: "f32[128, 28, 28, 256]" = torch.ops.prims.convert_element_type.default(view_778, torch.float32);  view_778 = None
        sub_160: "f32[128, 28, 28, 256]" = torch.ops.aten.sub.Tensor(convert_element_type_559, arg644_1);  convert_element_type_559 = arg644_1 = None
        mul_566: "f32[128, 28, 28, 256]" = torch.ops.aten.mul.Tensor(sub_160, arg645_1);  sub_160 = None
        mul_567: "f32[128, 28, 28, 256]" = torch.ops.aten.mul.Tensor(mul_564, mul_566);  mul_564 = None
        sum_318: "f32[128, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_567, [3], True);  mul_567 = None
        mul_568: "f32[128, 28, 28, 256]" = torch.ops.aten.mul.Tensor(mul_566, sum_318);  sum_318 = None
        sub_161: "f32[128, 28, 28, 256]" = torch.ops.aten.sub.Tensor(mul_565, sum_317);  mul_565 = sum_317 = None
        sub_162: "f32[128, 28, 28, 256]" = torch.ops.aten.sub.Tensor(sub_161, mul_568);  sub_161 = mul_568 = None
        div_113: "f32[128, 28, 28, 1]" = torch.ops.aten.div.Tensor(arg645_1, 256);  arg645_1 = None
        mul_569: "f32[128, 28, 28, 256]" = torch.ops.aten.mul.Tensor(div_113, sub_162);  div_113 = sub_162 = None
        mul_570: "f32[128, 28, 28, 256]" = torch.ops.aten.mul.Tensor(view_777, mul_566);  mul_566 = None
        sum_319: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_570, [0, 1, 2]);  mul_570 = None
        sum_320: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_777, [0, 1, 2]);  view_777 = None
        convert_element_type_560: "bf16[128, 28, 28, 256]" = torch.ops.prims.convert_element_type.default(mul_569, torch.bfloat16);  mul_569 = None
        add_121: "bf16[128, 28, 28, 256]" = torch.ops.aten.add.Tensor(view_752, convert_element_type_560);  view_752 = convert_element_type_560 = None
        view_779: "bf16[100352, 256]" = torch.ops.aten.view.default(add_121, _shape_param_761);  add_121 = _shape_param_761 = None
        permute_246: "bf16[256, 100352]" = torch.ops.aten.permute.default(view_779, [1, 0])
        mm_181: "bf16[256, 512]" = torch.ops.aten.mm.default(permute_246, arg646_1);  permute_246 = arg646_1 = None
        mm_182: "bf16[100352, 512]" = torch.ops.aten.mm.default(view_779, arg647_1);  view_779 = arg647_1 = None
        view_780: "bf16[128, 28, 28, 512]" = torch.ops.aten.view.default(mm_182, _shape_param_762);  mm_182 = _shape_param_762 = None
        convert_element_type_561: "f32[128, 28, 28, 512]" = torch.ops.prims.convert_element_type.default(view_780, torch.float32);  view_780 = None
        convert_element_type_562: "f32[256, 512]" = torch.ops.prims.convert_element_type.default(mm_181, torch.float32);  mm_181 = None
        mul_571: "f32[128, 28, 28, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_561, arg648_1);  arg648_1 = None
        mul_572: "f32[128, 28, 28, 512]" = torch.ops.aten.mul.Tensor(mul_571, 512)
        sum_321: "f32[128, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_571, [3], True)
        mul_573: "f32[128, 28, 28, 512]" = torch.ops.aten.mul.Tensor(mul_571, arg649_1);  mul_571 = None
        sum_322: "f32[128, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_573, [3], True);  mul_573 = None
        mul_574: "f32[128, 28, 28, 512]" = torch.ops.aten.mul.Tensor(arg649_1, sum_322);  sum_322 = None
        sub_163: "f32[128, 28, 28, 512]" = torch.ops.aten.sub.Tensor(mul_572, sum_321);  mul_572 = sum_321 = None
        sub_164: "f32[128, 28, 28, 512]" = torch.ops.aten.sub.Tensor(sub_163, mul_574);  sub_163 = mul_574 = None
        mul_575: "f32[128, 28, 28, 512]" = torch.ops.aten.mul.Tensor(arg650_1, sub_164);  arg650_1 = sub_164 = None
        mul_576: "f32[128, 28, 28, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_561, arg649_1);  arg649_1 = None
        sum_323: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_576, [0, 1, 2]);  mul_576 = None
        sum_324: "f32[512]" = torch.ops.aten.sum.dim_IntList(convert_element_type_561, [0, 1, 2]);  convert_element_type_561 = None
        view_781: "f32[128, 28, 28, 2, 2, 128]" = torch.ops.aten.view.default(mul_575, _shape_param_763);  mul_575 = _shape_param_763 = None
        permute_247: "f32[128, 28, 2, 28, 2, 128]" = torch.ops.aten.permute.default(view_781, [0, 1, 4, 2, 3, 5]);  view_781 = None
        clone_108: "f32[128, 28, 2, 28, 2, 128]" = torch.ops.aten.clone.default(permute_247, memory_format = torch.contiguous_format);  permute_247 = None
        view_782: "f32[128, 56, 56, 128]" = torch.ops.aten.view.default(clone_108, _shape_param_764);  clone_108 = _shape_param_764 = None
        view_783: "f32[128, 3136, 128]" = torch.ops.aten.view.default(view_782, _shape_param_765);  view_782 = _shape_param_765 = None
        convert_element_type_563: "bf16[128, 3136, 128]" = torch.ops.prims.convert_element_type.default(view_783, torch.bfloat16)
        convert_element_type_564: "bf16[128, 1, 1]" = torch.ops.prims.convert_element_type.default(arg651_1, torch.bfloat16);  arg651_1 = None
        div_114: "bf16[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_564, 0.9956521736457944);  convert_element_type_564 = None
        mul_577: "bf16[128, 3136, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_563, div_114);  convert_element_type_563 = div_114 = None
        view_784: "bf16[401408, 128]" = torch.ops.aten.view.default(mul_577, _shape_param_766);  mul_577 = _shape_param_766 = None
        mm_183: "bf16[401408, 512]" = torch.ops.aten.mm.default(view_784, arg652_1);  arg652_1 = None
        permute_248: "bf16[128, 401408]" = torch.ops.aten.permute.default(view_784, [1, 0])
        mm_184: "bf16[128, 512]" = torch.ops.aten.mm.default(permute_248, arg653_1);  permute_248 = arg653_1 = None
        sum_325: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_784, [0], True, dtype = torch.float32);  view_784 = None
        view_785: "f32[128]" = torch.ops.aten.view.default(sum_325, _shape_param_767);  sum_325 = _shape_param_767 = None
        convert_element_type_565: "bf16[128]" = torch.ops.prims.convert_element_type.default(view_785, torch.bfloat16);  view_785 = None
        view_786: "bf16[128, 3136, 512]" = torch.ops.aten.view.default(mm_183, _shape_param_768);  mm_183 = _shape_param_768 = None
        convert_element_type_566: "f32[128, 512]" = torch.ops.prims.convert_element_type.default(mm_184, torch.float32);  mm_184 = None
        convert_element_type_567: "f32[128]" = torch.ops.prims.convert_element_type.default(convert_element_type_565, torch.float32);  convert_element_type_565 = None
        convert_element_type_568: "f32[128, 3136, 512]" = torch.ops.prims.convert_element_type.default(view_786, torch.float32);  view_786 = None
        view_787: "bf16[128, 3136, 512]" = torch.ops.aten.view.default(arg654_1, _shape_param_769);  arg654_1 = _shape_param_769 = None
        convert_element_type_569: "f32[128, 3136, 512]" = torch.ops.prims.convert_element_type.default(view_787, torch.float32);  view_787 = None
        mul_578: "f32[128, 3136, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_569, 0.7071067811865476)
        erf_22: "f32[128, 3136, 512]" = torch.ops.aten.erf.default(mul_578);  mul_578 = None
        add_122: "f32[128, 3136, 512]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
        mul_579: "f32[128, 3136, 512]" = torch.ops.aten.mul.Tensor(add_122, 0.5);  add_122 = None
        mul_580: "f32[128, 3136, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_569, convert_element_type_569)
        mul_581: "f32[128, 3136, 512]" = torch.ops.aten.mul.Tensor(mul_580, -0.5);  mul_580 = None
        exp_44: "f32[128, 3136, 512]" = torch.ops.aten.exp.default(mul_581);  mul_581 = None
        mul_582: "f32[128, 3136, 512]" = torch.ops.aten.mul.Tensor(exp_44, 0.3989422804014327);  exp_44 = None
        mul_583: "f32[128, 3136, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_569, mul_582);  convert_element_type_569 = mul_582 = None
        add_123: "f32[128, 3136, 512]" = torch.ops.aten.add.Tensor(mul_579, mul_583);  mul_579 = mul_583 = None
        mul_584: "f32[128, 3136, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_568, add_123);  convert_element_type_568 = add_123 = None
        convert_element_type_570: "bf16[128, 3136, 512]" = torch.ops.prims.convert_element_type.default(mul_584, torch.bfloat16);  mul_584 = None
        view_788: "bf16[401408, 512]" = torch.ops.aten.view.default(convert_element_type_570, _shape_param_770);  convert_element_type_570 = _shape_param_770 = None
        mm_185: "bf16[401408, 128]" = torch.ops.aten.mm.default(view_788, arg655_1);  arg655_1 = None
        permute_249: "bf16[512, 401408]" = torch.ops.aten.permute.default(view_788, [1, 0])
        mm_186: "bf16[512, 128]" = torch.ops.aten.mm.default(permute_249, arg656_1);  permute_249 = arg656_1 = None
        sum_326: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_788, [0], True, dtype = torch.float32);  view_788 = None
        view_789: "f32[512]" = torch.ops.aten.view.default(sum_326, _shape_param_771);  sum_326 = _shape_param_771 = None
        convert_element_type_571: "bf16[512]" = torch.ops.prims.convert_element_type.default(view_789, torch.bfloat16);  view_789 = None
        view_790: "bf16[128, 3136, 128]" = torch.ops.aten.view.default(mm_185, _shape_param_772);  mm_185 = _shape_param_772 = None
        convert_element_type_572: "f32[128, 3136, 128]" = torch.ops.prims.convert_element_type.default(view_790, torch.float32);  view_790 = None
        convert_element_type_573: "f32[512, 128]" = torch.ops.prims.convert_element_type.default(mm_186, torch.float32);  mm_186 = None
        convert_element_type_574: "f32[512]" = torch.ops.prims.convert_element_type.default(convert_element_type_571, torch.float32);  convert_element_type_571 = None
        mul_585: "f32[128, 3136, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_572, arg657_1);  arg657_1 = None
        mul_586: "f32[128, 3136, 128]" = torch.ops.aten.mul.Tensor(mul_585, 128)
        sum_327: "f32[128, 3136, 1]" = torch.ops.aten.sum.dim_IntList(mul_585, [2], True)
        mul_587: "f32[128, 3136, 128]" = torch.ops.aten.mul.Tensor(mul_585, arg658_1);  mul_585 = None
        sum_328: "f32[128, 3136, 1]" = torch.ops.aten.sum.dim_IntList(mul_587, [2], True);  mul_587 = None
        mul_588: "f32[128, 3136, 128]" = torch.ops.aten.mul.Tensor(arg658_1, sum_328);  sum_328 = None
        sub_165: "f32[128, 3136, 128]" = torch.ops.aten.sub.Tensor(mul_586, sum_327);  mul_586 = sum_327 = None
        sub_166: "f32[128, 3136, 128]" = torch.ops.aten.sub.Tensor(sub_165, mul_588);  sub_165 = mul_588 = None
        mul_589: "f32[128, 3136, 128]" = torch.ops.aten.mul.Tensor(arg659_1, sub_166);  arg659_1 = sub_166 = None
        mul_590: "f32[128, 3136, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_572, arg658_1);  arg658_1 = None
        sum_329: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_590, [0, 1]);  mul_590 = None
        sum_330: "f32[128]" = torch.ops.aten.sum.dim_IntList(convert_element_type_572, [0, 1]);  convert_element_type_572 = None
        add_124: "f32[128, 3136, 128]" = torch.ops.aten.add.Tensor(view_783, mul_589);  view_783 = mul_589 = None
        view_791: "f32[128, 56, 56, 128]" = torch.ops.aten.view.default(add_124, _shape_param_773);  add_124 = _shape_param_773 = None
        convert_element_type_575: "bf16[128, 56, 56, 128]" = torch.ops.prims.convert_element_type.default(view_791, torch.bfloat16)
        convert_element_type_576: "bf16[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(arg660_1, torch.bfloat16);  arg660_1 = None
        div_115: "bf16[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_576, 0.9956521736457944);  convert_element_type_576 = None
        mul_591: "bf16[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_575, div_115);  convert_element_type_575 = div_115 = None
        iota_2: "i64[56]" = torch.ops.prims.iota.default(56, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_125: "i64[56]" = torch.ops.aten.add.Tensor(iota_2, 3);  iota_2 = None
        fmod_2: "i64[56]" = torch.ops.aten.fmod.Scalar(add_125, 56);  add_125 = None
        index_62: "bf16[128, 56, 56, 128]" = torch.ops.aten.index.Tensor(mul_591, [None, None, fmod_2]);  mul_591 = None
        index_63: "bf16[128, 56, 56, 128]" = torch.ops.aten.index.Tensor(index_62, [None, fmod_2]);  index_62 = fmod_2 = None
        view_792: "bf16[128, 8, 7, 8, 7, 128]" = torch.ops.aten.view.default(index_63, _shape_param_774);  index_63 = _shape_param_774 = None
        permute_250: "bf16[128, 8, 8, 7, 7, 128]" = torch.ops.aten.permute.default(view_792, [0, 1, 3, 2, 4, 5]);  view_792 = None
        clone_109: "bf16[128, 8, 8, 7, 7, 128]" = torch.ops.aten.clone.default(permute_250, memory_format = torch.contiguous_format);  permute_250 = None
        view_793: "bf16[8192, 7, 7, 128]" = torch.ops.aten.view.default(clone_109, _shape_param_775);  clone_109 = _shape_param_775 = None
        view_794: "bf16[8192, 49, 128]" = torch.ops.aten.view.default(view_793, _shape_param_776);  view_793 = _shape_param_776 = None
        view_795: "bf16[401408, 128]" = torch.ops.aten.view.default(view_794, _shape_param_777);  view_794 = _shape_param_777 = None
        mm_187: "bf16[401408, 128]" = torch.ops.aten.mm.default(view_795, arg661_1);  arg661_1 = None
        permute_251: "bf16[128, 401408]" = torch.ops.aten.permute.default(view_795, [1, 0])
        mm_188: "bf16[128, 128]" = torch.ops.aten.mm.default(permute_251, arg662_1);  permute_251 = arg662_1 = None
        sum_331: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_795, [0], True, dtype = torch.float32);  view_795 = None
        view_796: "f32[128]" = torch.ops.aten.view.default(sum_331, _shape_param_778);  sum_331 = _shape_param_778 = None
        convert_element_type_577: "bf16[128]" = torch.ops.prims.convert_element_type.default(view_796, torch.bfloat16);  view_796 = None
        view_797: "bf16[8192, 49, 128]" = torch.ops.aten.view.default(mm_187, _shape_param_779);  mm_187 = _shape_param_779 = None
        convert_element_type_578: "f32[128, 128]" = torch.ops.prims.convert_element_type.default(mm_188, torch.float32);  mm_188 = None
        convert_element_type_579: "f32[128]" = torch.ops.prims.convert_element_type.default(convert_element_type_577, torch.float32);  convert_element_type_577 = None
        view_798: "bf16[8192, 49, 4, 32]" = torch.ops.aten.view.default(view_797, _shape_param_780);  view_797 = _shape_param_780 = None
        permute_252: "bf16[8192, 4, 49, 32]" = torch.ops.aten.permute.default(view_798, [0, 2, 1, 3]);  view_798 = None
        clone_110: "bf16[8192, 4, 49, 32]" = torch.ops.aten.clone.default(permute_252, memory_format = torch.contiguous_format);  permute_252 = None
        view_799: "bf16[32768, 49, 32]" = torch.ops.aten.view.default(clone_110, _shape_param_781);  clone_110 = _shape_param_781 = None
        bmm_88: "bf16[32768, 49, 32]" = torch.ops.aten.bmm.default(arg663_1, view_799);  arg663_1 = None
        bmm_89: "bf16[32768, 49, 49]" = torch.ops.aten.bmm.default(view_799, arg664_1);  view_799 = arg664_1 = None
        view_800: "bf16[8192, 4, 49, 32]" = torch.ops.aten.view.default(bmm_88, _shape_param_782);  bmm_88 = _shape_param_782 = None
        view_801: "bf16[8192, 4, 49, 49]" = torch.ops.aten.view.default(bmm_89, _shape_param_783);  bmm_89 = _shape_param_783 = None
        convert_element_type_580: "f32[8192, 4, 49, 49]" = torch.ops.prims.convert_element_type.default(view_801, torch.float32);  view_801 = None
        slice_41: "bf16[32768, 49, 56]" = torch.ops.aten.slice.Tensor(arg665_1, 1, 0, -7);  arg665_1 = None
        slice_42: "bf16[32768, 49, 49]" = torch.ops.aten.slice.Tensor(slice_41, 2, 0, -7);  slice_41 = None
        view_802: "bf16[8192, 4, 49, 49]" = torch.ops.aten.view.default(slice_42, _shape_param_784);  slice_42 = _shape_param_784 = None
        view_803: "i64[2401]" = torch.ops.aten.view.default(arg666_1, [-1]);  arg666_1 = None
        index_64: "f32[2401, 4]" = torch.ops.aten.index.Tensor(arg667_1, [view_803]);  arg667_1 = None
        view_804: "f32[49, 49, 4]" = torch.ops.aten.view.default(index_64, _shape_param_785);  index_64 = _shape_param_785 = None
        permute_253: "f32[4, 49, 49]" = torch.ops.aten.permute.default(view_804, [2, 0, 1]);  view_804 = None
        clone_111: "f32[4, 49, 49]" = torch.ops.aten.clone.default(permute_253, memory_format = torch.contiguous_format);  permute_253 = None
        unsqueeze_44: "f32[1, 4, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_111, 0);  clone_111 = None
        add_126: "f32[8192, 4, 49, 49]" = torch.ops.aten.add.Tensor(view_802, unsqueeze_44);  view_802 = unsqueeze_44 = None
        view_805: "f32[128, 64, 4, 49, 49]" = torch.ops.aten.view.default(add_126, _shape_param_786);  add_126 = _shape_param_786 = None
        unsqueeze_45: "f32[64, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(arg668_1, 1);  arg668_1 = None
        unsqueeze_46: "f32[1, 64, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_45, 0);  unsqueeze_45 = None
        add_127: "f32[128, 64, 4, 49, 49]" = torch.ops.aten.add.Tensor(view_805, unsqueeze_46);  view_805 = unsqueeze_46 = None
        view_806: "f32[8192, 4, 49, 49]" = torch.ops.aten.view.default(add_127, _shape_param_787);  add_127 = _shape_param_787 = None
        sub_167: "f32[8192, 4, 49, 49]" = torch.ops.aten.sub.Tensor(view_806, arg669_1);  view_806 = arg669_1 = None
        exp_45: "f32[8192, 4, 49, 49]" = torch.ops.aten.exp.default(sub_167);  sub_167 = None
        div_116: "f32[8192, 4, 49, 49]" = torch.ops.aten.div.Tensor(exp_45, arg670_1);  exp_45 = arg670_1 = None
        mul_592: "f32[8192, 4, 49, 49]" = torch.ops.aten.mul.Tensor(convert_element_type_580, div_116);  convert_element_type_580 = None
        sum_332: "f32[8192, 4, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_592, [-1], True)
        neg_22: "f32[8192, 4, 49, 49]" = torch.ops.aten.neg.default(div_116);  div_116 = None
        fma_22: "f32[8192, 4, 49, 49]" = torch.ops.prims.fma.default(neg_22, sum_332, mul_592);  neg_22 = sum_332 = mul_592 = None
        convert_element_type_581: "bf16[8192, 4, 49, 49]" = torch.ops.prims.convert_element_type.default(fma_22, torch.bfloat16)
        sum_333: "f32[1, 4, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_22, [0], True, dtype = torch.float32);  fma_22 = None
        squeeze_22: "f32[4, 49, 49]" = torch.ops.aten.squeeze.dim(sum_333, 0);  sum_333 = None
        permute_254: "f32[49, 49, 4]" = torch.ops.aten.permute.default(squeeze_22, [1, 2, 0]);  squeeze_22 = None
        view_807: "f32[2401, 4]" = torch.ops.aten.view.default(permute_254, _shape_param_788);  permute_254 = _shape_param_788 = None
        full_3: "f32[169, 4]" = torch.ops.aten.full.default(_shape_param_789, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_789 = None
        index_put_22: "f32[169, 4]" = torch.ops.aten.index_put.default(full_3, [view_803], view_807, True);  view_803 = view_807 = None
        view_808: "bf16[32768, 49, 49]" = torch.ops.aten.view.default(convert_element_type_581, _shape_param_790);  convert_element_type_581 = _shape_param_790 = None
        bmm_90: "bf16[32768, 32, 49]" = torch.ops.aten.bmm.default(arg671_1, view_808);  arg671_1 = None
        bmm_91: "bf16[32768, 49, 32]" = torch.ops.aten.bmm.default(view_808, arg672_1);  view_808 = arg672_1 = None
        view_809: "bf16[8192, 4, 32, 49]" = torch.ops.aten.view.default(bmm_90, _shape_param_791);  bmm_90 = _shape_param_791 = None
        view_810: "bf16[8192, 4, 49, 32]" = torch.ops.aten.view.default(bmm_91, _shape_param_792);  bmm_91 = _shape_param_792 = None
        permute_255: "bf16[8192, 4, 49, 32]" = torch.ops.aten.permute.default(view_809, [0, 1, 3, 2]);  view_809 = None
        mul_593: "bf16[8192, 4, 49, 32]" = torch.ops.aten.mul.Tensor(view_810, 0.1767766952966369);  view_810 = None
        cat_22: "bf16[24576, 4, 49, 32]" = torch.ops.aten.cat.default([mul_593, permute_255, view_800]);  mul_593 = permute_255 = view_800 = None
        view_811: "bf16[3, 8192, 4, 49, 32]" = torch.ops.aten.view.default(cat_22, _shape_param_793);  cat_22 = _shape_param_793 = None
        permute_256: "bf16[8192, 49, 3, 4, 32]" = torch.ops.aten.permute.default(view_811, [1, 3, 0, 2, 4]);  view_811 = None
        clone_112: "bf16[8192, 49, 3, 4, 32]" = torch.ops.aten.clone.default(permute_256, memory_format = torch.contiguous_format);  permute_256 = None
        view_812: "bf16[8192, 49, 384]" = torch.ops.aten.view.default(clone_112, _shape_param_794);  clone_112 = _shape_param_794 = None
        view_813: "bf16[401408, 384]" = torch.ops.aten.view.default(view_812, _shape_param_795);  view_812 = _shape_param_795 = None
        mm_189: "bf16[401408, 128]" = torch.ops.aten.mm.default(view_813, arg673_1);  arg673_1 = None
        permute_257: "bf16[384, 401408]" = torch.ops.aten.permute.default(view_813, [1, 0])
        mm_190: "bf16[384, 128]" = torch.ops.aten.mm.default(permute_257, arg674_1);  permute_257 = arg674_1 = None
        sum_334: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_813, [0], True, dtype = torch.float32);  view_813 = None
        view_814: "f32[384]" = torch.ops.aten.view.default(sum_334, _shape_param_796);  sum_334 = _shape_param_796 = None
        convert_element_type_582: "bf16[384]" = torch.ops.prims.convert_element_type.default(view_814, torch.bfloat16);  view_814 = None
        view_815: "bf16[8192, 49, 128]" = torch.ops.aten.view.default(mm_189, _shape_param_797);  mm_189 = _shape_param_797 = None
        convert_element_type_583: "f32[8192, 49, 128]" = torch.ops.prims.convert_element_type.default(view_815, torch.float32);  view_815 = None
        convert_element_type_584: "f32[384, 128]" = torch.ops.prims.convert_element_type.default(mm_190, torch.float32);  mm_190 = None
        convert_element_type_585: "f32[384]" = torch.ops.prims.convert_element_type.default(convert_element_type_582, torch.float32);  convert_element_type_582 = None
        view_816: "f32[8192, 7, 7, 128]" = torch.ops.aten.view.default(convert_element_type_583, _shape_param_798);  convert_element_type_583 = _shape_param_798 = None
        view_817: "f32[128, 8, 8, 7, 7, 128]" = torch.ops.aten.view.default(view_816, _shape_param_799);  view_816 = _shape_param_799 = None
        permute_258: "f32[128, 8, 7, 8, 7, 128]" = torch.ops.aten.permute.default(view_817, [0, 1, 3, 2, 4, 5]);  view_817 = None
        clone_113: "f32[128, 8, 7, 8, 7, 128]" = torch.ops.aten.clone.default(permute_258, memory_format = torch.contiguous_format);  permute_258 = None
        view_818: "f32[128, 56, 56, 128]" = torch.ops.aten.view.default(clone_113, _shape_param_800);  clone_113 = _shape_param_800 = None
        index_65: "f32[128, 56, 56, 128]" = torch.ops.aten.index.Tensor(view_818, [None, None, arg675_1]);  view_818 = None
        index_66: "f32[128, 56, 56, 128]" = torch.ops.aten.index.Tensor(index_65, [None, arg675_1]);  index_65 = arg675_1 = None
        mul_594: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(index_66, arg676_1);  arg676_1 = None
        mul_595: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_594, 128)
        sum_335: "f32[128, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_594, [3], True)
        mul_596: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_594, arg677_1);  mul_594 = None
        sum_336: "f32[128, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_596, [3], True);  mul_596 = None
        mul_597: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(arg677_1, sum_336);  sum_336 = None
        sub_168: "f32[128, 56, 56, 128]" = torch.ops.aten.sub.Tensor(mul_595, sum_335);  mul_595 = sum_335 = None
        sub_169: "f32[128, 56, 56, 128]" = torch.ops.aten.sub.Tensor(sub_168, mul_597);  sub_168 = mul_597 = None
        mul_598: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(arg678_1, sub_169);  arg678_1 = sub_169 = None
        mul_599: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(index_66, arg677_1);  arg677_1 = None
        sum_337: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_599, [0, 1, 2]);  mul_599 = None
        sum_338: "f32[128]" = torch.ops.aten.sum.dim_IntList(index_66, [0, 1, 2]);  index_66 = None
        add_128: "f32[128, 56, 56, 128]" = torch.ops.aten.add.Tensor(view_791, mul_598);  view_791 = mul_598 = None
        view_819: "f32[128, 3136, 128]" = torch.ops.aten.view.default(add_128, _shape_param_801);  add_128 = _shape_param_801 = None
        convert_element_type_586: "bf16[128, 3136, 128]" = torch.ops.prims.convert_element_type.default(view_819, torch.bfloat16)
        view_820: "bf16[401408, 128]" = torch.ops.aten.view.default(convert_element_type_586, _shape_param_802);  convert_element_type_586 = _shape_param_802 = None
        mm_191: "bf16[401408, 512]" = torch.ops.aten.mm.default(view_820, arg679_1);  arg679_1 = None
        permute_259: "bf16[128, 401408]" = torch.ops.aten.permute.default(view_820, [1, 0])
        mm_192: "bf16[128, 512]" = torch.ops.aten.mm.default(permute_259, arg680_1);  permute_259 = arg680_1 = None
        sum_339: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_820, [0], True, dtype = torch.float32);  view_820 = None
        view_821: "f32[128]" = torch.ops.aten.view.default(sum_339, _shape_param_803);  sum_339 = _shape_param_803 = None
        convert_element_type_587: "bf16[128]" = torch.ops.prims.convert_element_type.default(view_821, torch.bfloat16);  view_821 = None
        view_822: "bf16[128, 3136, 512]" = torch.ops.aten.view.default(mm_191, _shape_param_804);  mm_191 = _shape_param_804 = None
        convert_element_type_588: "f32[128, 512]" = torch.ops.prims.convert_element_type.default(mm_192, torch.float32);  mm_192 = None
        convert_element_type_589: "f32[128]" = torch.ops.prims.convert_element_type.default(convert_element_type_587, torch.float32);  convert_element_type_587 = None
        convert_element_type_590: "f32[128, 3136, 512]" = torch.ops.prims.convert_element_type.default(view_822, torch.float32);  view_822 = None
        view_823: "bf16[128, 3136, 512]" = torch.ops.aten.view.default(arg681_1, _shape_param_805);  arg681_1 = _shape_param_805 = None
        convert_element_type_591: "f32[128, 3136, 512]" = torch.ops.prims.convert_element_type.default(view_823, torch.float32);  view_823 = None
        mul_600: "f32[128, 3136, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_591, 0.7071067811865476)
        erf_23: "f32[128, 3136, 512]" = torch.ops.aten.erf.default(mul_600);  mul_600 = None
        add_129: "f32[128, 3136, 512]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
        mul_601: "f32[128, 3136, 512]" = torch.ops.aten.mul.Tensor(add_129, 0.5);  add_129 = None
        mul_602: "f32[128, 3136, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_591, convert_element_type_591)
        mul_603: "f32[128, 3136, 512]" = torch.ops.aten.mul.Tensor(mul_602, -0.5);  mul_602 = None
        exp_46: "f32[128, 3136, 512]" = torch.ops.aten.exp.default(mul_603);  mul_603 = None
        mul_604: "f32[128, 3136, 512]" = torch.ops.aten.mul.Tensor(exp_46, 0.3989422804014327);  exp_46 = None
        mul_605: "f32[128, 3136, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_591, mul_604);  convert_element_type_591 = mul_604 = None
        add_130: "f32[128, 3136, 512]" = torch.ops.aten.add.Tensor(mul_601, mul_605);  mul_601 = mul_605 = None
        mul_606: "f32[128, 3136, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_590, add_130);  convert_element_type_590 = add_130 = None
        convert_element_type_592: "bf16[128, 3136, 512]" = torch.ops.prims.convert_element_type.default(mul_606, torch.bfloat16);  mul_606 = None
        view_824: "bf16[401408, 512]" = torch.ops.aten.view.default(convert_element_type_592, _shape_param_806);  convert_element_type_592 = _shape_param_806 = None
        mm_193: "bf16[401408, 128]" = torch.ops.aten.mm.default(view_824, arg682_1);  arg682_1 = None
        permute_260: "bf16[512, 401408]" = torch.ops.aten.permute.default(view_824, [1, 0])
        mm_194: "bf16[512, 128]" = torch.ops.aten.mm.default(permute_260, arg683_1);  permute_260 = arg683_1 = None
        sum_340: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_824, [0], True, dtype = torch.float32);  view_824 = None
        view_825: "f32[512]" = torch.ops.aten.view.default(sum_340, _shape_param_807);  sum_340 = _shape_param_807 = None
        convert_element_type_593: "bf16[512]" = torch.ops.prims.convert_element_type.default(view_825, torch.bfloat16);  view_825 = None
        view_826: "bf16[128, 3136, 128]" = torch.ops.aten.view.default(mm_193, _shape_param_808);  mm_193 = _shape_param_808 = None
        convert_element_type_594: "f32[128, 3136, 128]" = torch.ops.prims.convert_element_type.default(view_826, torch.float32);  view_826 = None
        convert_element_type_595: "f32[512, 128]" = torch.ops.prims.convert_element_type.default(mm_194, torch.float32);  mm_194 = None
        convert_element_type_596: "f32[512]" = torch.ops.prims.convert_element_type.default(convert_element_type_593, torch.float32);  convert_element_type_593 = None
        mul_607: "f32[128, 3136, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_594, arg684_1);  arg684_1 = None
        mul_608: "f32[128, 3136, 128]" = torch.ops.aten.mul.Tensor(mul_607, 128)
        sum_341: "f32[128, 3136, 1]" = torch.ops.aten.sum.dim_IntList(mul_607, [2], True)
        mul_609: "f32[128, 3136, 128]" = torch.ops.aten.mul.Tensor(mul_607, arg685_1);  mul_607 = None
        sum_342: "f32[128, 3136, 1]" = torch.ops.aten.sum.dim_IntList(mul_609, [2], True);  mul_609 = None
        mul_610: "f32[128, 3136, 128]" = torch.ops.aten.mul.Tensor(arg685_1, sum_342);  sum_342 = None
        sub_170: "f32[128, 3136, 128]" = torch.ops.aten.sub.Tensor(mul_608, sum_341);  mul_608 = sum_341 = None
        sub_171: "f32[128, 3136, 128]" = torch.ops.aten.sub.Tensor(sub_170, mul_610);  sub_170 = mul_610 = None
        mul_611: "f32[128, 3136, 128]" = torch.ops.aten.mul.Tensor(arg686_1, sub_171);  arg686_1 = sub_171 = None
        mul_612: "f32[128, 3136, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_594, arg685_1);  arg685_1 = None
        sum_343: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_612, [0, 1]);  mul_612 = None
        sum_344: "f32[128]" = torch.ops.aten.sum.dim_IntList(convert_element_type_594, [0, 1]);  convert_element_type_594 = None
        add_131: "f32[128, 3136, 128]" = torch.ops.aten.add.Tensor(view_819, mul_611);  view_819 = mul_611 = None
        view_827: "f32[128, 56, 56, 128]" = torch.ops.aten.view.default(add_131, _shape_param_809);  add_131 = _shape_param_809 = None
        convert_element_type_597: "bf16[128, 56, 56, 128]" = torch.ops.prims.convert_element_type.default(view_827, torch.bfloat16)
        view_828: "bf16[128, 8, 7, 8, 7, 128]" = torch.ops.aten.view.default(convert_element_type_597, _shape_param_810);  convert_element_type_597 = _shape_param_810 = None
        permute_261: "bf16[128, 8, 8, 7, 7, 128]" = torch.ops.aten.permute.default(view_828, [0, 1, 3, 2, 4, 5]);  view_828 = None
        clone_114: "bf16[128, 8, 8, 7, 7, 128]" = torch.ops.aten.clone.default(permute_261, memory_format = torch.contiguous_format);  permute_261 = None
        view_829: "bf16[8192, 7, 7, 128]" = torch.ops.aten.view.default(clone_114, _shape_param_811);  clone_114 = _shape_param_811 = None
        view_830: "bf16[8192, 49, 128]" = torch.ops.aten.view.default(view_829, _shape_param_812);  view_829 = _shape_param_812 = None
        view_831: "bf16[401408, 128]" = torch.ops.aten.view.default(view_830, _shape_param_813);  view_830 = _shape_param_813 = None
        mm_195: "bf16[401408, 128]" = torch.ops.aten.mm.default(view_831, arg687_1);  arg687_1 = None
        permute_262: "bf16[128, 401408]" = torch.ops.aten.permute.default(view_831, [1, 0])
        mm_196: "bf16[128, 128]" = torch.ops.aten.mm.default(permute_262, arg688_1);  permute_262 = arg688_1 = None
        sum_345: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_831, [0], True, dtype = torch.float32);  view_831 = None
        view_832: "f32[128]" = torch.ops.aten.view.default(sum_345, _shape_param_814);  sum_345 = _shape_param_814 = None
        convert_element_type_598: "bf16[128]" = torch.ops.prims.convert_element_type.default(view_832, torch.bfloat16);  view_832 = None
        view_833: "bf16[8192, 49, 128]" = torch.ops.aten.view.default(mm_195, _shape_param_815);  mm_195 = _shape_param_815 = None
        convert_element_type_599: "f32[128, 128]" = torch.ops.prims.convert_element_type.default(mm_196, torch.float32);  mm_196 = None
        convert_element_type_600: "f32[128]" = torch.ops.prims.convert_element_type.default(convert_element_type_598, torch.float32);  convert_element_type_598 = None
        view_834: "bf16[8192, 49, 4, 32]" = torch.ops.aten.view.default(view_833, _shape_param_816);  view_833 = _shape_param_816 = None
        permute_263: "bf16[8192, 4, 49, 32]" = torch.ops.aten.permute.default(view_834, [0, 2, 1, 3]);  view_834 = None
        clone_115: "bf16[8192, 4, 49, 32]" = torch.ops.aten.clone.default(permute_263, memory_format = torch.contiguous_format);  permute_263 = None
        view_835: "bf16[32768, 49, 32]" = torch.ops.aten.view.default(clone_115, _shape_param_817);  clone_115 = _shape_param_817 = None
        bmm_92: "bf16[32768, 49, 32]" = torch.ops.aten.bmm.default(arg689_1, view_835);  arg689_1 = None
        bmm_93: "bf16[32768, 49, 49]" = torch.ops.aten.bmm.default(view_835, arg690_1);  view_835 = arg690_1 = None
        view_836: "bf16[8192, 4, 49, 32]" = torch.ops.aten.view.default(bmm_92, _shape_param_818);  bmm_92 = _shape_param_818 = None
        view_837: "bf16[8192, 4, 49, 49]" = torch.ops.aten.view.default(bmm_93, _shape_param_819);  bmm_93 = _shape_param_819 = None
        convert_element_type_601: "f32[8192, 4, 49, 49]" = torch.ops.prims.convert_element_type.default(view_837, torch.float32);  view_837 = None
        slice_43: "bf16[32768, 49, 56]" = torch.ops.aten.slice.Tensor(arg691_1, 1, 0, -7);  arg691_1 = None
        slice_44: "bf16[32768, 49, 49]" = torch.ops.aten.slice.Tensor(slice_43, 2, 0, -7);  slice_43 = None
        view_838: "bf16[8192, 4, 49, 49]" = torch.ops.aten.view.default(slice_44, _shape_param_820);  slice_44 = _shape_param_820 = None
        view_839: "i64[2401]" = torch.ops.aten.view.default(arg692_1, [-1]);  arg692_1 = None
        index_67: "f32[2401, 4]" = torch.ops.aten.index.Tensor(arg693_1, [view_839]);  arg693_1 = None
        view_840: "f32[49, 49, 4]" = torch.ops.aten.view.default(index_67, _shape_param_821);  index_67 = _shape_param_821 = None
        permute_264: "f32[4, 49, 49]" = torch.ops.aten.permute.default(view_840, [2, 0, 1]);  view_840 = None
        clone_116: "f32[4, 49, 49]" = torch.ops.aten.clone.default(permute_264, memory_format = torch.contiguous_format);  permute_264 = None
        unsqueeze_47: "f32[1, 4, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_116, 0);  clone_116 = None
        add_132: "f32[8192, 4, 49, 49]" = torch.ops.aten.add.Tensor(view_838, unsqueeze_47);  view_838 = unsqueeze_47 = None
        sub_172: "f32[8192, 4, 49, 49]" = torch.ops.aten.sub.Tensor(add_132, arg694_1);  add_132 = arg694_1 = None
        exp_47: "f32[8192, 4, 49, 49]" = torch.ops.aten.exp.default(sub_172);  sub_172 = None
        div_117: "f32[8192, 4, 49, 49]" = torch.ops.aten.div.Tensor(exp_47, arg695_1);  exp_47 = arg695_1 = None
        mul_613: "f32[8192, 4, 49, 49]" = torch.ops.aten.mul.Tensor(convert_element_type_601, div_117);  convert_element_type_601 = None
        sum_346: "f32[8192, 4, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_613, [-1], True)
        neg_23: "f32[8192, 4, 49, 49]" = torch.ops.aten.neg.default(div_117);  div_117 = None
        fma_23: "f32[8192, 4, 49, 49]" = torch.ops.prims.fma.default(neg_23, sum_346, mul_613);  neg_23 = sum_346 = mul_613 = None
        convert_element_type_602: "bf16[8192, 4, 49, 49]" = torch.ops.prims.convert_element_type.default(fma_23, torch.bfloat16)
        sum_347: "f32[1, 4, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_23, [0], True, dtype = torch.float32);  fma_23 = None
        squeeze_23: "f32[4, 49, 49]" = torch.ops.aten.squeeze.dim(sum_347, 0);  sum_347 = None
        permute_265: "f32[49, 49, 4]" = torch.ops.aten.permute.default(squeeze_23, [1, 2, 0]);  squeeze_23 = None
        view_841: "f32[2401, 4]" = torch.ops.aten.view.default(permute_265, _shape_param_822);  permute_265 = _shape_param_822 = None
        index_put_23: "f32[169, 4]" = torch.ops.aten.index_put.default(full_3, [view_839], view_841, True);  full_3 = view_839 = view_841 = None
        view_842: "bf16[32768, 49, 49]" = torch.ops.aten.view.default(convert_element_type_602, _shape_param_823);  convert_element_type_602 = _shape_param_823 = None
        bmm_94: "bf16[32768, 32, 49]" = torch.ops.aten.bmm.default(arg696_1, view_842);  arg696_1 = None
        bmm_95: "bf16[32768, 49, 32]" = torch.ops.aten.bmm.default(view_842, arg697_1);  view_842 = arg697_1 = None
        view_843: "bf16[8192, 4, 32, 49]" = torch.ops.aten.view.default(bmm_94, _shape_param_824);  bmm_94 = _shape_param_824 = None
        view_844: "bf16[8192, 4, 49, 32]" = torch.ops.aten.view.default(bmm_95, _shape_param_825);  bmm_95 = _shape_param_825 = None
        permute_266: "bf16[8192, 4, 49, 32]" = torch.ops.aten.permute.default(view_843, [0, 1, 3, 2]);  view_843 = None
        mul_614: "bf16[8192, 4, 49, 32]" = torch.ops.aten.mul.Tensor(view_844, 0.1767766952966369);  view_844 = None
        cat_23: "bf16[24576, 4, 49, 32]" = torch.ops.aten.cat.default([mul_614, permute_266, view_836]);  mul_614 = permute_266 = view_836 = None
        view_845: "bf16[3, 8192, 4, 49, 32]" = torch.ops.aten.view.default(cat_23, _shape_param_826);  cat_23 = _shape_param_826 = None
        permute_267: "bf16[8192, 49, 3, 4, 32]" = torch.ops.aten.permute.default(view_845, [1, 3, 0, 2, 4]);  view_845 = None
        clone_117: "bf16[8192, 49, 3, 4, 32]" = torch.ops.aten.clone.default(permute_267, memory_format = torch.contiguous_format);  permute_267 = None
        view_846: "bf16[8192, 49, 384]" = torch.ops.aten.view.default(clone_117, _shape_param_827);  clone_117 = _shape_param_827 = None
        view_847: "bf16[401408, 384]" = torch.ops.aten.view.default(view_846, _shape_param_828);  view_846 = _shape_param_828 = None
        mm_197: "bf16[401408, 128]" = torch.ops.aten.mm.default(view_847, arg698_1);  arg698_1 = None
        permute_268: "bf16[384, 401408]" = torch.ops.aten.permute.default(view_847, [1, 0])
        mm_198: "bf16[384, 128]" = torch.ops.aten.mm.default(permute_268, arg699_1);  permute_268 = arg699_1 = None
        sum_348: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_847, [0], True, dtype = torch.float32);  view_847 = None
        view_848: "f32[384]" = torch.ops.aten.view.default(sum_348, _shape_param_829);  sum_348 = _shape_param_829 = None
        convert_element_type_603: "bf16[384]" = torch.ops.prims.convert_element_type.default(view_848, torch.bfloat16);  view_848 = None
        view_849: "bf16[8192, 49, 128]" = torch.ops.aten.view.default(mm_197, _shape_param_830);  mm_197 = _shape_param_830 = None
        convert_element_type_604: "f32[8192, 49, 128]" = torch.ops.prims.convert_element_type.default(view_849, torch.float32);  view_849 = None
        convert_element_type_605: "f32[384, 128]" = torch.ops.prims.convert_element_type.default(mm_198, torch.float32);  mm_198 = None
        convert_element_type_606: "f32[384]" = torch.ops.prims.convert_element_type.default(convert_element_type_603, torch.float32);  convert_element_type_603 = None
        view_850: "f32[8192, 7, 7, 128]" = torch.ops.aten.view.default(convert_element_type_604, _shape_param_831);  convert_element_type_604 = _shape_param_831 = None
        view_851: "f32[128, 8, 8, 7, 7, 128]" = torch.ops.aten.view.default(view_850, _shape_param_832);  view_850 = _shape_param_832 = None
        permute_269: "f32[128, 8, 7, 8, 7, 128]" = torch.ops.aten.permute.default(view_851, [0, 1, 3, 2, 4, 5]);  view_851 = None
        clone_118: "f32[128, 8, 7, 8, 7, 128]" = torch.ops.aten.clone.default(permute_269, memory_format = torch.contiguous_format);  permute_269 = None
        view_852: "f32[128, 56, 56, 128]" = torch.ops.aten.view.default(clone_118, _shape_param_833);  clone_118 = _shape_param_833 = None
        mul_615: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(view_852, arg700_1);  arg700_1 = None
        mul_616: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_615, 128)
        sum_349: "f32[128, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_615, [3], True)
        permute_270: "bf16[128, 56, 56, 128]" = torch.ops.aten.permute.default(arg701_1, [0, 2, 3, 1]);  arg701_1 = None
        convert_element_type_607: "f32[128, 56, 56, 128]" = torch.ops.prims.convert_element_type.default(permute_270, torch.float32);  permute_270 = None
        sub_173: "f32[128, 56, 56, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_607, arg702_1);  convert_element_type_607 = arg702_1 = None
        mul_617: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(sub_173, arg703_1);  sub_173 = None
        mul_618: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_617, arg704_1)
        add_133: "f32[128, 56, 56, 128]" = torch.ops.aten.add.Tensor(mul_618, arg705_1);  mul_618 = arg705_1 = None
        sub_174: "f32[128, 56, 56, 128]" = torch.ops.aten.sub.Tensor(add_133, arg706_1);  add_133 = arg706_1 = None
        mul_619: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(sub_174, arg707_1);  sub_174 = None
        mul_620: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_615, mul_619);  mul_615 = None
        sum_350: "f32[128, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_620, [3], True);  mul_620 = None
        mul_621: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_619, sum_350);  sum_350 = None
        sub_175: "f32[128, 56, 56, 128]" = torch.ops.aten.sub.Tensor(mul_616, sum_349);  mul_616 = sum_349 = None
        sub_176: "f32[128, 56, 56, 128]" = torch.ops.aten.sub.Tensor(sub_175, mul_621);  sub_175 = mul_621 = None
        div_118: "f32[128, 56, 56, 1]" = torch.ops.aten.div.Tensor(arg707_1, 128);  arg707_1 = None
        mul_622: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(div_118, sub_176);  div_118 = sub_176 = None
        mul_623: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(view_852, mul_619);  mul_619 = None
        sum_351: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_623, [0, 1, 2]);  mul_623 = None
        sum_352: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_852, [0, 1, 2]);  view_852 = None
        add_134: "f32[128, 56, 56, 128]" = torch.ops.aten.add.Tensor(view_827, mul_622);  view_827 = mul_622 = None
        mul_624: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(add_134, arg704_1);  arg704_1 = None
        mul_625: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_624, 128)
        sum_353: "f32[128, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_624, [3], True)
        mul_626: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_624, mul_617);  mul_624 = None
        sum_354: "f32[128, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_626, [3], True);  mul_626 = None
        mul_627: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_617, sum_354);  sum_354 = None
        sub_177: "f32[128, 56, 56, 128]" = torch.ops.aten.sub.Tensor(mul_625, sum_353);  mul_625 = sum_353 = None
        sub_178: "f32[128, 56, 56, 128]" = torch.ops.aten.sub.Tensor(sub_177, mul_627);  sub_177 = mul_627 = None
        div_119: "f32[128, 56, 56, 1]" = torch.ops.aten.div.Tensor(arg703_1, 128);  arg703_1 = None
        mul_628: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(div_119, sub_178);  div_119 = sub_178 = None
        mul_629: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(add_134, mul_617);  mul_617 = None
        sum_355: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_629, [0, 1, 2]);  mul_629 = None
        sum_356: "f32[128]" = torch.ops.aten.sum.dim_IntList(add_134, [0, 1, 2]);  add_134 = None
        convert_element_type_608: "bf16[128, 56, 56, 128]" = torch.ops.prims.convert_element_type.default(mul_628, torch.bfloat16);  mul_628 = None
        permute_271: "bf16[128, 128, 56, 56]" = torch.ops.aten.permute.default(convert_element_type_608, [0, 3, 1, 2]);  convert_element_type_608 = None
        sum_357: "bf16[128]" = torch.ops.aten.sum.dim_IntList(permute_271, [0, 2, 3])
        convert_element_type_609: "f32[128]" = torch.ops.prims.convert_element_type.default(sum_357, torch.float32);  sum_357 = None
        return (sum_3, sum_4, convert_element_type_5, convert_element_type_6, convert_element_type_12, convert_element_type_13, sum_9, sum_10, convert_element_type_18, convert_element_type_19, index_put, convert_element_type_24, convert_element_type_25, sum_17, sum_18, convert_element_type_30, convert_element_type_31, convert_element_type_37, convert_element_type_38, sum_23, sum_24, convert_element_type_43, convert_element_type_44, index_put_1, convert_element_type_49, convert_element_type_50, sum_31, sum_32, convert_element_type_54, sum_35, sum_36, convert_element_type_59, convert_element_type_60, convert_element_type_66, convert_element_type_67, sum_41, sum_42, convert_element_type_72, convert_element_type_73, index_put_2, convert_element_type_78, convert_element_type_79, sum_49, sum_50, convert_element_type_84, convert_element_type_85, convert_element_type_91, convert_element_type_92, sum_55, sum_56, convert_element_type_97, convert_element_type_98, index_put_3, convert_element_type_103, convert_element_type_104, sum_63, sum_64, convert_element_type_109, convert_element_type_110, convert_element_type_116, convert_element_type_117, sum_69, sum_70, convert_element_type_122, convert_element_type_123, index_put_4, convert_element_type_128, convert_element_type_129, sum_77, sum_78, convert_element_type_134, convert_element_type_135, convert_element_type_141, convert_element_type_142, sum_83, sum_84, convert_element_type_147, convert_element_type_148, index_put_5, convert_element_type_153, convert_element_type_154, sum_91, sum_92, convert_element_type_159, convert_element_type_160, convert_element_type_166, convert_element_type_167, sum_97, sum_98, convert_element_type_172, convert_element_type_173, index_put_6, convert_element_type_178, convert_element_type_179, sum_105, sum_106, convert_element_type_184, convert_element_type_185, convert_element_type_191, convert_element_type_192, sum_111, sum_112, convert_element_type_197, convert_element_type_198, index_put_7, convert_element_type_203, convert_element_type_204, sum_119, sum_120, convert_element_type_209, convert_element_type_210, convert_element_type_216, convert_element_type_217, sum_125, sum_126, convert_element_type_222, convert_element_type_223, index_put_8, convert_element_type_228, convert_element_type_229, sum_133, sum_134, convert_element_type_234, convert_element_type_235, convert_element_type_241, convert_element_type_242, sum_139, sum_140, convert_element_type_247, convert_element_type_248, index_put_9, convert_element_type_253, convert_element_type_254, sum_147, sum_148, convert_element_type_259, convert_element_type_260, convert_element_type_266, convert_element_type_267, sum_153, sum_154, convert_element_type_272, convert_element_type_273, index_put_10, convert_element_type_278, convert_element_type_279, sum_161, sum_162, convert_element_type_284, convert_element_type_285, convert_element_type_291, convert_element_type_292, sum_167, sum_168, convert_element_type_297, convert_element_type_298, index_put_11, convert_element_type_303, convert_element_type_304, sum_175, sum_176, convert_element_type_309, convert_element_type_310, convert_element_type_316, convert_element_type_317, sum_181, sum_182, convert_element_type_322, convert_element_type_323, index_put_12, convert_element_type_328, convert_element_type_329, sum_189, sum_190, convert_element_type_334, convert_element_type_335, convert_element_type_341, convert_element_type_342, sum_195, sum_196, convert_element_type_347, convert_element_type_348, index_put_13, convert_element_type_353, convert_element_type_354, sum_203, sum_204, convert_element_type_359, convert_element_type_360, convert_element_type_366, convert_element_type_367, sum_209, sum_210, convert_element_type_372, convert_element_type_373, index_put_14, convert_element_type_378, convert_element_type_379, sum_217, sum_218, convert_element_type_384, convert_element_type_385, convert_element_type_391, convert_element_type_392, sum_223, sum_224, convert_element_type_397, convert_element_type_398, index_put_15, convert_element_type_403, convert_element_type_404, sum_231, sum_232, convert_element_type_409, convert_element_type_410, convert_element_type_416, convert_element_type_417, sum_237, sum_238, convert_element_type_422, convert_element_type_423, index_put_16, convert_element_type_428, convert_element_type_429, sum_245, sum_246, convert_element_type_434, convert_element_type_435, convert_element_type_441, convert_element_type_442, sum_251, sum_252, convert_element_type_447, convert_element_type_448, index_put_17, convert_element_type_453, convert_element_type_454, sum_259, sum_260, convert_element_type_459, convert_element_type_460, convert_element_type_466, convert_element_type_467, sum_265, sum_266, convert_element_type_472, convert_element_type_473, index_put_18, convert_element_type_478, convert_element_type_479, sum_273, sum_274, convert_element_type_484, convert_element_type_485, convert_element_type_491, convert_element_type_492, sum_279, sum_280, convert_element_type_497, convert_element_type_498, index_put_19, convert_element_type_503, convert_element_type_504, sum_287, sum_288, convert_element_type_508, sum_291, sum_292, convert_element_type_513, convert_element_type_514, convert_element_type_520, convert_element_type_521, sum_297, sum_298, convert_element_type_526, convert_element_type_527, index_put_20, convert_element_type_532, convert_element_type_533, sum_305, sum_306, convert_element_type_538, convert_element_type_539, convert_element_type_545, convert_element_type_546, sum_311, sum_312, convert_element_type_551, convert_element_type_552, index_put_21, convert_element_type_557, convert_element_type_558, sum_319, sum_320, convert_element_type_562, sum_323, sum_324, convert_element_type_566, convert_element_type_567, convert_element_type_573, convert_element_type_574, sum_329, sum_330, convert_element_type_578, convert_element_type_579, index_put_22, convert_element_type_584, convert_element_type_585, sum_337, sum_338, convert_element_type_588, convert_element_type_589, convert_element_type_595, convert_element_type_596, sum_343, sum_344, convert_element_type_599, convert_element_type_600, index_put_23, convert_element_type_605, convert_element_type_606, sum_351, sum_352, sum_355, sum_356, permute_271, convert_element_type_609)



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
