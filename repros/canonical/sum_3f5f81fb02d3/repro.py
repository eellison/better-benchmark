"""
Standalone repro captured via capture_hook.
Label: torchbench_moco_infer
Pattern hash: 3f5f81fb02d3
Shape hash: a1c343e2
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([32, 128], f32), T([64, 3, 7, 7], f32), T([64, 3, 7, 7], f32), T([64], f32), T([64], f32), T([64], f32), T([64], f32), T([64, 64, 1, 1], f32), T([64, 64, 1, 1], f32), T([64], f32), T([64], f32), T([64], f32), T([64], f32), T([64, 64, 3, 3], f32), T([64, 64, 3, 3], f32), T([64], f32), T([64], f32), T([64], f32), T([64], f32), T([256, 64, 1, 1], f32), T([256, 64, 1, 1], f32), T([256], f32), T([256], f32), T([256], f32), T([256], f32), T([256, 64, 1, 1], f32), T([256, 64, 1, 1], f32), T([256], f32), T([256], f32), T([256], f32), T([256], f32), T([64, 256, 1, 1], f32), T([64, 256, 1, 1], f32), T([64], f32), T([64], f32), T([64], f32), T([64], f32), T([64, 64, 3, 3], f32), T([64, 64, 3, 3], f32), T([64], f32), T([64], f32), T([64], f32), T([64], f32), T([256, 64, 1, 1], f32), T([256, 64, 1, 1], f32), T([256], f32), T([256], f32), T([256], f32), T([256], f32), T([64, 256, 1, 1], f32), T([64, 256, 1, 1], f32), T([64], f32), T([64], f32), T([64], f32), T([64], f32), T([64, 64, 3, 3], f32), T([64, 64, 3, 3], f32), T([64], f32), T([64], f32), T([64], f32), T([64], f32), T([256, 64, 1, 1], f32), T([256, 64, 1, 1], f32), T([256], f32), T([256], f32), T([256], f32), T([256], f32), T([128, 256, 1, 1], f32), T([128, 256, 1, 1], f32), T([128], f32), T([128], f32), T([128], f32), T([128], f32), T([128, 128, 3, 3], f32), T([128, 128, 3, 3], f32), T([128], f32), T([128], f32), T([128], f32), T([128], f32), T([512, 128, 1, 1], f32), T([512, 128, 1, 1], f32), T([512], f32), T([512], f32), T([512], f32), T([512], f32), T([512, 256, 1, 1], f32), T([512, 256, 1, 1], f32), T([512], f32), T([512], f32), T([512], f32), T([512], f32), T([128, 512, 1, 1], f32), T([128, 512, 1, 1], f32), T([128], f32), T([128], f32), T([128], f32), T([128], f32), T([128, 128, 3, 3], f32), T([128, 128, 3, 3], f32), T([128], f32), T([128], f32), T([128], f32), T([128], f32), T([512, 128, 1, 1], f32), T([512, 128, 1, 1], f32), T([512], f32), T([512], f32), T([512], f32), T([512], f32), T([128, 512, 1, 1], f32), T([128, 512, 1, 1], f32), T([128], f32), T([128], f32), T([128], f32), T([128], f32), T([128, 128, 3, 3], f32), T([128, 128, 3, 3], f32), T([128], f32), T([128], f32), T([128], f32), T([128], f32), T([512, 128, 1, 1], f32), T([512, 128, 1, 1], f32), T([512], f32), T([512], f32), T([512], f32), T([512], f32), T([128, 512, 1, 1], f32), T([128, 512, 1, 1], f32), T([128], f32), T([128], f32), T([128], f32), T([128], f32), T([128, 128, 3, 3], f32), T([128, 128, 3, 3], f32), T([128], f32), T([128], f32), T([128], f32), T([128], f32), T([512, 128, 1, 1], f32), T([512, 128, 1, 1], f32), T([512], f32), T([512], f32), T([512], f32), T([512], f32), T([256, 512, 1, 1], f32), T([256, 512, 1, 1], f32), T([256], f32), T([256], f32), T([256], f32), T([256], f32), T([256, 256, 3, 3], f32), T([256, 256, 3, 3], f32), T([256], f32), T([256], f32), T([256], f32), T([256], f32), T([1024, 256, 1, 1], f32), T([1024, 256, 1, 1], f32), T([1024], f32), T([1024], f32), T([1024], f32), T([1024], f32), T([1024, 512, 1, 1], f32), T([1024, 512, 1, 1], f32), T([1024], f32), T([1024], f32), T([1024], f32), T([1024], f32), T([256, 1024, 1, 1], f32), T([256, 1024, 1, 1], f32), T([256], f32), T([256], f32), T([256], f32), T([256], f32), T([256, 256, 3, 3], f32), T([256, 256, 3, 3], f32), T([256], f32), T([256], f32), T([256], f32), T([256], f32), T([1024, 256, 1, 1], f32), T([1024, 256, 1, 1], f32), T([1024], f32), T([1024], f32), T([1024], f32), T([1024], f32), T([256, 1024, 1, 1], f32), T([256, 1024, 1, 1], f32), T([256], f32), T([256], f32), T([256], f32), T([256], f32), T([256, 256, 3, 3], f32), T([256, 256, 3, 3], f32), T([256], f32), T([256], f32), T([256], f32), T([256], f32), T([1024, 256, 1, 1], f32), T([1024, 256, 1, 1], f32), T([1024], f32), T([1024], f32), T([1024], f32), T([1024], f32), T([256, 1024, 1, 1], f32), T([256, 1024, 1, 1], f32), T([256], f32), T([256], f32), T([256], f32), T([256], f32), T([256, 256, 3, 3], f32), T([256, 256, 3, 3], f32), T([256], f32), T([256], f32), T([256], f32), T([256], f32), T([1024, 256, 1, 1], f32), T([1024, 256, 1, 1], f32), T([1024], f32), T([1024], f32), T([1024], f32), T([1024], f32), T([256, 1024, 1, 1], f32), T([256, 1024, 1, 1], f32), T([256], f32), T([256], f32), T([256], f32), T([256], f32), T([256, 256, 3, 3], f32), T([256, 256, 3, 3], f32), T([256], f32), T([256], f32), T([256], f32), T([256], f32), T([1024, 256, 1, 1], f32), T([1024, 256, 1, 1], f32), T([1024], f32), T([1024], f32), T([1024], f32), T([1024], f32), T([256, 1024, 1, 1], f32), T([256, 1024, 1, 1], f32), T([256], f32), T([256], f32), T([256], f32), T([256], f32), T([256, 256, 3, 3], f32), T([256, 256, 3, 3], f32), T([256], f32), T([256], f32), T([256], f32), T([256], f32), T([1024, 256, 1, 1], f32), T([1024, 256, 1, 1], f32), T([1024], f32), T([1024], f32), T([1024], f32), T([1024], f32), T([512, 1024, 1, 1], f32), T([512, 1024, 1, 1], f32), T([512], f32), T([512], f32), T([512], f32), T([512], f32), T([512, 512, 3, 3], f32), T([512, 512, 3, 3], f32), T([512], f32), T([512], f32), T([512], f32), T([512], f32), T([2048, 512, 1, 1], f32), T([2048, 512, 1, 1], f32), T([2048], f32), T([2048], f32), T([2048], f32), T([2048], f32), T([2048, 1024, 1, 1], f32), T([2048, 1024, 1, 1], f32), T([2048], f32), T([2048], f32), T([2048], f32), T([2048], f32), T([512, 2048, 1, 1], f32), T([512, 2048, 1, 1], f32), T([512], f32), T([512], f32), T([512], f32), T([512], f32), T([512, 512, 3, 3], f32), T([512, 512, 3, 3], f32), T([512], f32), T([512], f32), T([512], f32), T([512], f32), T([2048, 512, 1, 1], f32), T([2048, 512, 1, 1], f32), T([2048], f32), T([2048], f32), T([2048], f32), T([2048], f32), T([512, 2048, 1, 1], f32), T([512, 2048, 1, 1], f32), T([512], f32), T([512], f32), T([512], f32), T([512], f32), T([512, 512, 3, 3], f32), T([512, 512, 3, 3], f32), T([512], f32), T([512], f32), T([512], f32), T([512], f32), T([2048, 512, 1, 1], f32), T([2048, 512, 1, 1], f32), T([2048], f32), T([2048], f32), T([2048], f32), T([2048], f32), T([128, 2048], f32), T([128, 2048], f32), T([128], f32), T([128], f32), S([32, 128]))"

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "f32[32, 128]", arg1_1: "f32[64, 3, 7, 7]", arg2_1: "f32[64, 3, 7, 7]", arg3_1: "f32[64]", arg4_1: "f32[64]", arg5_1: "f32[64]", arg6_1: "f32[64]", arg7_1: "f32[64, 64, 1, 1]", arg8_1: "f32[64, 64, 1, 1]", arg9_1: "f32[64]", arg10_1: "f32[64]", arg11_1: "f32[64]", arg12_1: "f32[64]", arg13_1: "f32[64, 64, 3, 3]", arg14_1: "f32[64, 64, 3, 3]", arg15_1: "f32[64]", arg16_1: "f32[64]", arg17_1: "f32[64]", arg18_1: "f32[64]", arg19_1: "f32[256, 64, 1, 1]", arg20_1: "f32[256, 64, 1, 1]", arg21_1: "f32[256]", arg22_1: "f32[256]", arg23_1: "f32[256]", arg24_1: "f32[256]", arg25_1: "f32[256, 64, 1, 1]", arg26_1: "f32[256, 64, 1, 1]", arg27_1: "f32[256]", arg28_1: "f32[256]", arg29_1: "f32[256]", arg30_1: "f32[256]", arg31_1: "f32[64, 256, 1, 1]", arg32_1: "f32[64, 256, 1, 1]", arg33_1: "f32[64]", arg34_1: "f32[64]", arg35_1: "f32[64]", arg36_1: "f32[64]", arg37_1: "f32[64, 64, 3, 3]", arg38_1: "f32[64, 64, 3, 3]", arg39_1: "f32[64]", arg40_1: "f32[64]", arg41_1: "f32[64]", arg42_1: "f32[64]", arg43_1: "f32[256, 64, 1, 1]", arg44_1: "f32[256, 64, 1, 1]", arg45_1: "f32[256]", arg46_1: "f32[256]", arg47_1: "f32[256]", arg48_1: "f32[256]", arg49_1: "f32[64, 256, 1, 1]", arg50_1: "f32[64, 256, 1, 1]", arg51_1: "f32[64]", arg52_1: "f32[64]", arg53_1: "f32[64]", arg54_1: "f32[64]", arg55_1: "f32[64, 64, 3, 3]", arg56_1: "f32[64, 64, 3, 3]", arg57_1: "f32[64]", arg58_1: "f32[64]", arg59_1: "f32[64]", arg60_1: "f32[64]", arg61_1: "f32[256, 64, 1, 1]", arg62_1: "f32[256, 64, 1, 1]", arg63_1: "f32[256]", arg64_1: "f32[256]", arg65_1: "f32[256]", arg66_1: "f32[256]", arg67_1: "f32[128, 256, 1, 1]", arg68_1: "f32[128, 256, 1, 1]", arg69_1: "f32[128]", arg70_1: "f32[128]", arg71_1: "f32[128]", arg72_1: "f32[128]", arg73_1: "f32[128, 128, 3, 3]", arg74_1: "f32[128, 128, 3, 3]", arg75_1: "f32[128]", arg76_1: "f32[128]", arg77_1: "f32[128]", arg78_1: "f32[128]", arg79_1: "f32[512, 128, 1, 1]", arg80_1: "f32[512, 128, 1, 1]", arg81_1: "f32[512]", arg82_1: "f32[512]", arg83_1: "f32[512]", arg84_1: "f32[512]", arg85_1: "f32[512, 256, 1, 1]", arg86_1: "f32[512, 256, 1, 1]", arg87_1: "f32[512]", arg88_1: "f32[512]", arg89_1: "f32[512]", arg90_1: "f32[512]", arg91_1: "f32[128, 512, 1, 1]", arg92_1: "f32[128, 512, 1, 1]", arg93_1: "f32[128]", arg94_1: "f32[128]", arg95_1: "f32[128]", arg96_1: "f32[128]", arg97_1: "f32[128, 128, 3, 3]", arg98_1: "f32[128, 128, 3, 3]", arg99_1: "f32[128]", arg100_1: "f32[128]", arg101_1: "f32[128]", arg102_1: "f32[128]", arg103_1: "f32[512, 128, 1, 1]", arg104_1: "f32[512, 128, 1, 1]", arg105_1: "f32[512]", arg106_1: "f32[512]", arg107_1: "f32[512]", arg108_1: "f32[512]", arg109_1: "f32[128, 512, 1, 1]", arg110_1: "f32[128, 512, 1, 1]", arg111_1: "f32[128]", arg112_1: "f32[128]", arg113_1: "f32[128]", arg114_1: "f32[128]", arg115_1: "f32[128, 128, 3, 3]", arg116_1: "f32[128, 128, 3, 3]", arg117_1: "f32[128]", arg118_1: "f32[128]", arg119_1: "f32[128]", arg120_1: "f32[128]", arg121_1: "f32[512, 128, 1, 1]", arg122_1: "f32[512, 128, 1, 1]", arg123_1: "f32[512]", arg124_1: "f32[512]", arg125_1: "f32[512]", arg126_1: "f32[512]", arg127_1: "f32[128, 512, 1, 1]", arg128_1: "f32[128, 512, 1, 1]", arg129_1: "f32[128]", arg130_1: "f32[128]", arg131_1: "f32[128]", arg132_1: "f32[128]", arg133_1: "f32[128, 128, 3, 3]", arg134_1: "f32[128, 128, 3, 3]", arg135_1: "f32[128]", arg136_1: "f32[128]", arg137_1: "f32[128]", arg138_1: "f32[128]", arg139_1: "f32[512, 128, 1, 1]", arg140_1: "f32[512, 128, 1, 1]", arg141_1: "f32[512]", arg142_1: "f32[512]", arg143_1: "f32[512]", arg144_1: "f32[512]", arg145_1: "f32[256, 512, 1, 1]", arg146_1: "f32[256, 512, 1, 1]", arg147_1: "f32[256]", arg148_1: "f32[256]", arg149_1: "f32[256]", arg150_1: "f32[256]", arg151_1: "f32[256, 256, 3, 3]", arg152_1: "f32[256, 256, 3, 3]", arg153_1: "f32[256]", arg154_1: "f32[256]", arg155_1: "f32[256]", arg156_1: "f32[256]", arg157_1: "f32[1024, 256, 1, 1]", arg158_1: "f32[1024, 256, 1, 1]", arg159_1: "f32[1024]", arg160_1: "f32[1024]", arg161_1: "f32[1024]", arg162_1: "f32[1024]", arg163_1: "f32[1024, 512, 1, 1]", arg164_1: "f32[1024, 512, 1, 1]", arg165_1: "f32[1024]", arg166_1: "f32[1024]", arg167_1: "f32[1024]", arg168_1: "f32[1024]", arg169_1: "f32[256, 1024, 1, 1]", arg170_1: "f32[256, 1024, 1, 1]", arg171_1: "f32[256]", arg172_1: "f32[256]", arg173_1: "f32[256]", arg174_1: "f32[256]", arg175_1: "f32[256, 256, 3, 3]", arg176_1: "f32[256, 256, 3, 3]", arg177_1: "f32[256]", arg178_1: "f32[256]", arg179_1: "f32[256]", arg180_1: "f32[256]", arg181_1: "f32[1024, 256, 1, 1]", arg182_1: "f32[1024, 256, 1, 1]", arg183_1: "f32[1024]", arg184_1: "f32[1024]", arg185_1: "f32[1024]", arg186_1: "f32[1024]", arg187_1: "f32[256, 1024, 1, 1]", arg188_1: "f32[256, 1024, 1, 1]", arg189_1: "f32[256]", arg190_1: "f32[256]", arg191_1: "f32[256]", arg192_1: "f32[256]", arg193_1: "f32[256, 256, 3, 3]", arg194_1: "f32[256, 256, 3, 3]", arg195_1: "f32[256]", arg196_1: "f32[256]", arg197_1: "f32[256]", arg198_1: "f32[256]", arg199_1: "f32[1024, 256, 1, 1]", arg200_1: "f32[1024, 256, 1, 1]", arg201_1: "f32[1024]", arg202_1: "f32[1024]", arg203_1: "f32[1024]", arg204_1: "f32[1024]", arg205_1: "f32[256, 1024, 1, 1]", arg206_1: "f32[256, 1024, 1, 1]", arg207_1: "f32[256]", arg208_1: "f32[256]", arg209_1: "f32[256]", arg210_1: "f32[256]", arg211_1: "f32[256, 256, 3, 3]", arg212_1: "f32[256, 256, 3, 3]", arg213_1: "f32[256]", arg214_1: "f32[256]", arg215_1: "f32[256]", arg216_1: "f32[256]", arg217_1: "f32[1024, 256, 1, 1]", arg218_1: "f32[1024, 256, 1, 1]", arg219_1: "f32[1024]", arg220_1: "f32[1024]", arg221_1: "f32[1024]", arg222_1: "f32[1024]", arg223_1: "f32[256, 1024, 1, 1]", arg224_1: "f32[256, 1024, 1, 1]", arg225_1: "f32[256]", arg226_1: "f32[256]", arg227_1: "f32[256]", arg228_1: "f32[256]", arg229_1: "f32[256, 256, 3, 3]", arg230_1: "f32[256, 256, 3, 3]", arg231_1: "f32[256]", arg232_1: "f32[256]", arg233_1: "f32[256]", arg234_1: "f32[256]", arg235_1: "f32[1024, 256, 1, 1]", arg236_1: "f32[1024, 256, 1, 1]", arg237_1: "f32[1024]", arg238_1: "f32[1024]", arg239_1: "f32[1024]", arg240_1: "f32[1024]", arg241_1: "f32[256, 1024, 1, 1]", arg242_1: "f32[256, 1024, 1, 1]", arg243_1: "f32[256]", arg244_1: "f32[256]", arg245_1: "f32[256]", arg246_1: "f32[256]", arg247_1: "f32[256, 256, 3, 3]", arg248_1: "f32[256, 256, 3, 3]", arg249_1: "f32[256]", arg250_1: "f32[256]", arg251_1: "f32[256]", arg252_1: "f32[256]", arg253_1: "f32[1024, 256, 1, 1]", arg254_1: "f32[1024, 256, 1, 1]", arg255_1: "f32[1024]", arg256_1: "f32[1024]", arg257_1: "f32[1024]", arg258_1: "f32[1024]", arg259_1: "f32[512, 1024, 1, 1]", arg260_1: "f32[512, 1024, 1, 1]", arg261_1: "f32[512]", arg262_1: "f32[512]", arg263_1: "f32[512]", arg264_1: "f32[512]", arg265_1: "f32[512, 512, 3, 3]", arg266_1: "f32[512, 512, 3, 3]", arg267_1: "f32[512]", arg268_1: "f32[512]", arg269_1: "f32[512]", arg270_1: "f32[512]", arg271_1: "f32[2048, 512, 1, 1]", arg272_1: "f32[2048, 512, 1, 1]", arg273_1: "f32[2048]", arg274_1: "f32[2048]", arg275_1: "f32[2048]", arg276_1: "f32[2048]", arg277_1: "f32[2048, 1024, 1, 1]", arg278_1: "f32[2048, 1024, 1, 1]", arg279_1: "f32[2048]", arg280_1: "f32[2048]", arg281_1: "f32[2048]", arg282_1: "f32[2048]", arg283_1: "f32[512, 2048, 1, 1]", arg284_1: "f32[512, 2048, 1, 1]", arg285_1: "f32[512]", arg286_1: "f32[512]", arg287_1: "f32[512]", arg288_1: "f32[512]", arg289_1: "f32[512, 512, 3, 3]", arg290_1: "f32[512, 512, 3, 3]", arg291_1: "f32[512]", arg292_1: "f32[512]", arg293_1: "f32[512]", arg294_1: "f32[512]", arg295_1: "f32[2048, 512, 1, 1]", arg296_1: "f32[2048, 512, 1, 1]", arg297_1: "f32[2048]", arg298_1: "f32[2048]", arg299_1: "f32[2048]", arg300_1: "f32[2048]", arg301_1: "f32[512, 2048, 1, 1]", arg302_1: "f32[512, 2048, 1, 1]", arg303_1: "f32[512]", arg304_1: "f32[512]", arg305_1: "f32[512]", arg306_1: "f32[512]", arg307_1: "f32[512, 512, 3, 3]", arg308_1: "f32[512, 512, 3, 3]", arg309_1: "f32[512]", arg310_1: "f32[512]", arg311_1: "f32[512]", arg312_1: "f32[512]", arg313_1: "f32[2048, 512, 1, 1]", arg314_1: "f32[2048, 512, 1, 1]", arg315_1: "f32[2048]", arg316_1: "f32[2048]", arg317_1: "f32[2048]", arg318_1: "f32[2048]", arg319_1: "f32[128, 2048]", arg320_1: "f32[128, 2048]", arg321_1: "f32[128]", arg322_1: "f32[128]", _shape_param_0):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/moco/moco/builder.py:135 in forward, code: q = nn.functional.normalize(q, dim=1)
        pow_tensor_scalar: "f32[32, 128]" = torch.ops.aten.pow.Tensor_Scalar(arg0_1, 2.0)
        sum_dim_int_list: "f32[32, 1]" = torch.ops.aten.sum.dim_IntList(pow_tensor_scalar, [1], True);  pow_tensor_scalar = None
        pow_tensor_scalar_1: "f32[32, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_dim_int_list, 0.5);  sum_dim_int_list = None
        clamp_min_default: "f32[32, 1]" = torch.ops.aten.clamp_min.default(pow_tensor_scalar_1, 1e-12);  pow_tensor_scalar_1 = None
        expand_default: "f32[32, 128]" = torch.ops.aten.expand.default(clamp_min_default, _shape_param_0);  clamp_min_default = _shape_param_0 = None
        div_tensor: "f32[32, 128]" = torch.ops.aten.div.Tensor(arg0_1, expand_default);  arg0_1 = expand_default = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/moco/moco/builder.py:59 in _momentum_update_key_encoder, code: param_k.mul_(self.m).add_(param_q.mul(1.0 - self.m))
        mul_tensor: "f32[64, 3, 7, 7]" = torch.ops.aten.mul.Tensor(arg1_1, 0.999)
        mul_tensor_1: "f32[64, 3, 7, 7]" = torch.ops.aten.mul.Tensor(arg2_1, 0.0010000000000000009);  arg2_1 = None
        add_tensor: "f32[64, 3, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor, mul_tensor_1);  mul_tensor = mul_tensor_1 = None
        mul_tensor_2: "f32[64]" = torch.ops.aten.mul.Tensor(arg3_1, 0.999)
        mul_tensor_3: "f32[64]" = torch.ops.aten.mul.Tensor(arg4_1, 0.0010000000000000009);  arg4_1 = None
        add_tensor_1: "f32[64]" = torch.ops.aten.add.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        mul_tensor_4: "f32[64]" = torch.ops.aten.mul.Tensor(arg5_1, 0.999)
        mul_tensor_5: "f32[64]" = torch.ops.aten.mul.Tensor(arg6_1, 0.0010000000000000009);  arg6_1 = None
        add_tensor_2: "f32[64]" = torch.ops.aten.add.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None
        mul_tensor_6: "f32[64, 64, 1, 1]" = torch.ops.aten.mul.Tensor(arg7_1, 0.999)
        mul_tensor_7: "f32[64, 64, 1, 1]" = torch.ops.aten.mul.Tensor(arg8_1, 0.0010000000000000009);  arg8_1 = None
        add_tensor_3: "f32[64, 64, 1, 1]" = torch.ops.aten.add.Tensor(mul_tensor_6, mul_tensor_7);  mul_tensor_6 = mul_tensor_7 = None
        mul_tensor_8: "f32[64]" = torch.ops.aten.mul.Tensor(arg9_1, 0.999)
        mul_tensor_9: "f32[64]" = torch.ops.aten.mul.Tensor(arg10_1, 0.0010000000000000009);  arg10_1 = None
        add_tensor_4: "f32[64]" = torch.ops.aten.add.Tensor(mul_tensor_8, mul_tensor_9);  mul_tensor_8 = mul_tensor_9 = None
        mul_tensor_10: "f32[64]" = torch.ops.aten.mul.Tensor(arg11_1, 0.999)
        mul_tensor_11: "f32[64]" = torch.ops.aten.mul.Tensor(arg12_1, 0.0010000000000000009);  arg12_1 = None
        add_tensor_5: "f32[64]" = torch.ops.aten.add.Tensor(mul_tensor_10, mul_tensor_11);  mul_tensor_10 = mul_tensor_11 = None
        mul_tensor_12: "f32[64, 64, 3, 3]" = torch.ops.aten.mul.Tensor(arg13_1, 0.999)
        mul_tensor_13: "f32[64, 64, 3, 3]" = torch.ops.aten.mul.Tensor(arg14_1, 0.0010000000000000009);  arg14_1 = None
        add_tensor_6: "f32[64, 64, 3, 3]" = torch.ops.aten.add.Tensor(mul_tensor_12, mul_tensor_13);  mul_tensor_12 = mul_tensor_13 = None
        mul_tensor_14: "f32[64]" = torch.ops.aten.mul.Tensor(arg15_1, 0.999)
        mul_tensor_15: "f32[64]" = torch.ops.aten.mul.Tensor(arg16_1, 0.0010000000000000009);  arg16_1 = None
        add_tensor_7: "f32[64]" = torch.ops.aten.add.Tensor(mul_tensor_14, mul_tensor_15);  mul_tensor_14 = mul_tensor_15 = None
        mul_tensor_16: "f32[64]" = torch.ops.aten.mul.Tensor(arg17_1, 0.999)
        mul_tensor_17: "f32[64]" = torch.ops.aten.mul.Tensor(arg18_1, 0.0010000000000000009);  arg18_1 = None
        add_tensor_8: "f32[64]" = torch.ops.aten.add.Tensor(mul_tensor_16, mul_tensor_17);  mul_tensor_16 = mul_tensor_17 = None
        mul_tensor_18: "f32[256, 64, 1, 1]" = torch.ops.aten.mul.Tensor(arg19_1, 0.999)
        mul_tensor_19: "f32[256, 64, 1, 1]" = torch.ops.aten.mul.Tensor(arg20_1, 0.0010000000000000009);  arg20_1 = None
        add_tensor_9: "f32[256, 64, 1, 1]" = torch.ops.aten.add.Tensor(mul_tensor_18, mul_tensor_19);  mul_tensor_18 = mul_tensor_19 = None
        mul_tensor_20: "f32[256]" = torch.ops.aten.mul.Tensor(arg21_1, 0.999)
        mul_tensor_21: "f32[256]" = torch.ops.aten.mul.Tensor(arg22_1, 0.0010000000000000009);  arg22_1 = None
        add_tensor_10: "f32[256]" = torch.ops.aten.add.Tensor(mul_tensor_20, mul_tensor_21);  mul_tensor_20 = mul_tensor_21 = None
        mul_tensor_22: "f32[256]" = torch.ops.aten.mul.Tensor(arg23_1, 0.999)
        mul_tensor_23: "f32[256]" = torch.ops.aten.mul.Tensor(arg24_1, 0.0010000000000000009);  arg24_1 = None
        add_tensor_11: "f32[256]" = torch.ops.aten.add.Tensor(mul_tensor_22, mul_tensor_23);  mul_tensor_22 = mul_tensor_23 = None
        mul_tensor_24: "f32[256, 64, 1, 1]" = torch.ops.aten.mul.Tensor(arg25_1, 0.999)
        mul_tensor_25: "f32[256, 64, 1, 1]" = torch.ops.aten.mul.Tensor(arg26_1, 0.0010000000000000009);  arg26_1 = None
        add_tensor_12: "f32[256, 64, 1, 1]" = torch.ops.aten.add.Tensor(mul_tensor_24, mul_tensor_25);  mul_tensor_24 = mul_tensor_25 = None
        mul_tensor_26: "f32[256]" = torch.ops.aten.mul.Tensor(arg27_1, 0.999)
        mul_tensor_27: "f32[256]" = torch.ops.aten.mul.Tensor(arg28_1, 0.0010000000000000009);  arg28_1 = None
        add_tensor_13: "f32[256]" = torch.ops.aten.add.Tensor(mul_tensor_26, mul_tensor_27);  mul_tensor_26 = mul_tensor_27 = None
        mul_tensor_28: "f32[256]" = torch.ops.aten.mul.Tensor(arg29_1, 0.999)
        mul_tensor_29: "f32[256]" = torch.ops.aten.mul.Tensor(arg30_1, 0.0010000000000000009);  arg30_1 = None
        add_tensor_14: "f32[256]" = torch.ops.aten.add.Tensor(mul_tensor_28, mul_tensor_29);  mul_tensor_28 = mul_tensor_29 = None
        mul_tensor_30: "f32[64, 256, 1, 1]" = torch.ops.aten.mul.Tensor(arg31_1, 0.999)
        mul_tensor_31: "f32[64, 256, 1, 1]" = torch.ops.aten.mul.Tensor(arg32_1, 0.0010000000000000009);  arg32_1 = None
        add_tensor_15: "f32[64, 256, 1, 1]" = torch.ops.aten.add.Tensor(mul_tensor_30, mul_tensor_31);  mul_tensor_30 = mul_tensor_31 = None
        mul_tensor_32: "f32[64]" = torch.ops.aten.mul.Tensor(arg33_1, 0.999)
        mul_tensor_33: "f32[64]" = torch.ops.aten.mul.Tensor(arg34_1, 0.0010000000000000009);  arg34_1 = None
        add_tensor_16: "f32[64]" = torch.ops.aten.add.Tensor(mul_tensor_32, mul_tensor_33);  mul_tensor_32 = mul_tensor_33 = None
        mul_tensor_34: "f32[64]" = torch.ops.aten.mul.Tensor(arg35_1, 0.999)
        mul_tensor_35: "f32[64]" = torch.ops.aten.mul.Tensor(arg36_1, 0.0010000000000000009);  arg36_1 = None
        add_tensor_17: "f32[64]" = torch.ops.aten.add.Tensor(mul_tensor_34, mul_tensor_35);  mul_tensor_34 = mul_tensor_35 = None
        mul_tensor_36: "f32[64, 64, 3, 3]" = torch.ops.aten.mul.Tensor(arg37_1, 0.999)
        mul_tensor_37: "f32[64, 64, 3, 3]" = torch.ops.aten.mul.Tensor(arg38_1, 0.0010000000000000009);  arg38_1 = None
        add_tensor_18: "f32[64, 64, 3, 3]" = torch.ops.aten.add.Tensor(mul_tensor_36, mul_tensor_37);  mul_tensor_36 = mul_tensor_37 = None
        mul_tensor_38: "f32[64]" = torch.ops.aten.mul.Tensor(arg39_1, 0.999)
        mul_tensor_39: "f32[64]" = torch.ops.aten.mul.Tensor(arg40_1, 0.0010000000000000009);  arg40_1 = None
        add_tensor_19: "f32[64]" = torch.ops.aten.add.Tensor(mul_tensor_38, mul_tensor_39);  mul_tensor_38 = mul_tensor_39 = None
        mul_tensor_40: "f32[64]" = torch.ops.aten.mul.Tensor(arg41_1, 0.999)
        mul_tensor_41: "f32[64]" = torch.ops.aten.mul.Tensor(arg42_1, 0.0010000000000000009);  arg42_1 = None
        add_tensor_20: "f32[64]" = torch.ops.aten.add.Tensor(mul_tensor_40, mul_tensor_41);  mul_tensor_40 = mul_tensor_41 = None
        mul_tensor_42: "f32[256, 64, 1, 1]" = torch.ops.aten.mul.Tensor(arg43_1, 0.999)
        mul_tensor_43: "f32[256, 64, 1, 1]" = torch.ops.aten.mul.Tensor(arg44_1, 0.0010000000000000009);  arg44_1 = None
        add_tensor_21: "f32[256, 64, 1, 1]" = torch.ops.aten.add.Tensor(mul_tensor_42, mul_tensor_43);  mul_tensor_42 = mul_tensor_43 = None
        mul_tensor_44: "f32[256]" = torch.ops.aten.mul.Tensor(arg45_1, 0.999)
        mul_tensor_45: "f32[256]" = torch.ops.aten.mul.Tensor(arg46_1, 0.0010000000000000009);  arg46_1 = None
        add_tensor_22: "f32[256]" = torch.ops.aten.add.Tensor(mul_tensor_44, mul_tensor_45);  mul_tensor_44 = mul_tensor_45 = None
        mul_tensor_46: "f32[256]" = torch.ops.aten.mul.Tensor(arg47_1, 0.999)
        mul_tensor_47: "f32[256]" = torch.ops.aten.mul.Tensor(arg48_1, 0.0010000000000000009);  arg48_1 = None
        add_tensor_23: "f32[256]" = torch.ops.aten.add.Tensor(mul_tensor_46, mul_tensor_47);  mul_tensor_46 = mul_tensor_47 = None
        mul_tensor_48: "f32[64, 256, 1, 1]" = torch.ops.aten.mul.Tensor(arg49_1, 0.999)
        mul_tensor_49: "f32[64, 256, 1, 1]" = torch.ops.aten.mul.Tensor(arg50_1, 0.0010000000000000009);  arg50_1 = None
        add_tensor_24: "f32[64, 256, 1, 1]" = torch.ops.aten.add.Tensor(mul_tensor_48, mul_tensor_49);  mul_tensor_48 = mul_tensor_49 = None
        mul_tensor_50: "f32[64]" = torch.ops.aten.mul.Tensor(arg51_1, 0.999)
        mul_tensor_51: "f32[64]" = torch.ops.aten.mul.Tensor(arg52_1, 0.0010000000000000009);  arg52_1 = None
        add_tensor_25: "f32[64]" = torch.ops.aten.add.Tensor(mul_tensor_50, mul_tensor_51);  mul_tensor_50 = mul_tensor_51 = None
        mul_tensor_52: "f32[64]" = torch.ops.aten.mul.Tensor(arg53_1, 0.999)
        mul_tensor_53: "f32[64]" = torch.ops.aten.mul.Tensor(arg54_1, 0.0010000000000000009);  arg54_1 = None
        add_tensor_26: "f32[64]" = torch.ops.aten.add.Tensor(mul_tensor_52, mul_tensor_53);  mul_tensor_52 = mul_tensor_53 = None
        mul_tensor_54: "f32[64, 64, 3, 3]" = torch.ops.aten.mul.Tensor(arg55_1, 0.999)
        mul_tensor_55: "f32[64, 64, 3, 3]" = torch.ops.aten.mul.Tensor(arg56_1, 0.0010000000000000009);  arg56_1 = None
        add_tensor_27: "f32[64, 64, 3, 3]" = torch.ops.aten.add.Tensor(mul_tensor_54, mul_tensor_55);  mul_tensor_54 = mul_tensor_55 = None
        mul_tensor_56: "f32[64]" = torch.ops.aten.mul.Tensor(arg57_1, 0.999)
        mul_tensor_57: "f32[64]" = torch.ops.aten.mul.Tensor(arg58_1, 0.0010000000000000009);  arg58_1 = None
        add_tensor_28: "f32[64]" = torch.ops.aten.add.Tensor(mul_tensor_56, mul_tensor_57);  mul_tensor_56 = mul_tensor_57 = None
        mul_tensor_58: "f32[64]" = torch.ops.aten.mul.Tensor(arg59_1, 0.999)
        mul_tensor_59: "f32[64]" = torch.ops.aten.mul.Tensor(arg60_1, 0.0010000000000000009);  arg60_1 = None
        add_tensor_29: "f32[64]" = torch.ops.aten.add.Tensor(mul_tensor_58, mul_tensor_59);  mul_tensor_58 = mul_tensor_59 = None
        mul_tensor_60: "f32[256, 64, 1, 1]" = torch.ops.aten.mul.Tensor(arg61_1, 0.999)
        mul_tensor_61: "f32[256, 64, 1, 1]" = torch.ops.aten.mul.Tensor(arg62_1, 0.0010000000000000009);  arg62_1 = None
        add_tensor_30: "f32[256, 64, 1, 1]" = torch.ops.aten.add.Tensor(mul_tensor_60, mul_tensor_61);  mul_tensor_60 = mul_tensor_61 = None
        mul_tensor_62: "f32[256]" = torch.ops.aten.mul.Tensor(arg63_1, 0.999)
        mul_tensor_63: "f32[256]" = torch.ops.aten.mul.Tensor(arg64_1, 0.0010000000000000009);  arg64_1 = None
        add_tensor_31: "f32[256]" = torch.ops.aten.add.Tensor(mul_tensor_62, mul_tensor_63);  mul_tensor_62 = mul_tensor_63 = None
        mul_tensor_64: "f32[256]" = torch.ops.aten.mul.Tensor(arg65_1, 0.999)
        mul_tensor_65: "f32[256]" = torch.ops.aten.mul.Tensor(arg66_1, 0.0010000000000000009);  arg66_1 = None
        add_tensor_32: "f32[256]" = torch.ops.aten.add.Tensor(mul_tensor_64, mul_tensor_65);  mul_tensor_64 = mul_tensor_65 = None
        mul_tensor_66: "f32[128, 256, 1, 1]" = torch.ops.aten.mul.Tensor(arg67_1, 0.999)
        mul_tensor_67: "f32[128, 256, 1, 1]" = torch.ops.aten.mul.Tensor(arg68_1, 0.0010000000000000009);  arg68_1 = None
        add_tensor_33: "f32[128, 256, 1, 1]" = torch.ops.aten.add.Tensor(mul_tensor_66, mul_tensor_67);  mul_tensor_66 = mul_tensor_67 = None
        mul_tensor_68: "f32[128]" = torch.ops.aten.mul.Tensor(arg69_1, 0.999)
        mul_tensor_69: "f32[128]" = torch.ops.aten.mul.Tensor(arg70_1, 0.0010000000000000009);  arg70_1 = None
        add_tensor_34: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_68, mul_tensor_69);  mul_tensor_68 = mul_tensor_69 = None
        mul_tensor_70: "f32[128]" = torch.ops.aten.mul.Tensor(arg71_1, 0.999)
        mul_tensor_71: "f32[128]" = torch.ops.aten.mul.Tensor(arg72_1, 0.0010000000000000009);  arg72_1 = None
        add_tensor_35: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_70, mul_tensor_71);  mul_tensor_70 = mul_tensor_71 = None
        mul_tensor_72: "f32[128, 128, 3, 3]" = torch.ops.aten.mul.Tensor(arg73_1, 0.999)
        mul_tensor_73: "f32[128, 128, 3, 3]" = torch.ops.aten.mul.Tensor(arg74_1, 0.0010000000000000009);  arg74_1 = None
        add_tensor_36: "f32[128, 128, 3, 3]" = torch.ops.aten.add.Tensor(mul_tensor_72, mul_tensor_73);  mul_tensor_72 = mul_tensor_73 = None
        mul_tensor_74: "f32[128]" = torch.ops.aten.mul.Tensor(arg75_1, 0.999)
        mul_tensor_75: "f32[128]" = torch.ops.aten.mul.Tensor(arg76_1, 0.0010000000000000009);  arg76_1 = None
        add_tensor_37: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_74, mul_tensor_75);  mul_tensor_74 = mul_tensor_75 = None
        mul_tensor_76: "f32[128]" = torch.ops.aten.mul.Tensor(arg77_1, 0.999)
        mul_tensor_77: "f32[128]" = torch.ops.aten.mul.Tensor(arg78_1, 0.0010000000000000009);  arg78_1 = None
        add_tensor_38: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_76, mul_tensor_77);  mul_tensor_76 = mul_tensor_77 = None
        mul_tensor_78: "f32[512, 128, 1, 1]" = torch.ops.aten.mul.Tensor(arg79_1, 0.999)
        mul_tensor_79: "f32[512, 128, 1, 1]" = torch.ops.aten.mul.Tensor(arg80_1, 0.0010000000000000009);  arg80_1 = None
        add_tensor_39: "f32[512, 128, 1, 1]" = torch.ops.aten.add.Tensor(mul_tensor_78, mul_tensor_79);  mul_tensor_78 = mul_tensor_79 = None
        mul_tensor_80: "f32[512]" = torch.ops.aten.mul.Tensor(arg81_1, 0.999)
        mul_tensor_81: "f32[512]" = torch.ops.aten.mul.Tensor(arg82_1, 0.0010000000000000009);  arg82_1 = None
        add_tensor_40: "f32[512]" = torch.ops.aten.add.Tensor(mul_tensor_80, mul_tensor_81);  mul_tensor_80 = mul_tensor_81 = None
        mul_tensor_82: "f32[512]" = torch.ops.aten.mul.Tensor(arg83_1, 0.999)
        mul_tensor_83: "f32[512]" = torch.ops.aten.mul.Tensor(arg84_1, 0.0010000000000000009);  arg84_1 = None
        add_tensor_41: "f32[512]" = torch.ops.aten.add.Tensor(mul_tensor_82, mul_tensor_83);  mul_tensor_82 = mul_tensor_83 = None
        mul_tensor_84: "f32[512, 256, 1, 1]" = torch.ops.aten.mul.Tensor(arg85_1, 0.999)
        mul_tensor_85: "f32[512, 256, 1, 1]" = torch.ops.aten.mul.Tensor(arg86_1, 0.0010000000000000009);  arg86_1 = None
        add_tensor_42: "f32[512, 256, 1, 1]" = torch.ops.aten.add.Tensor(mul_tensor_84, mul_tensor_85);  mul_tensor_84 = mul_tensor_85 = None
        mul_tensor_86: "f32[512]" = torch.ops.aten.mul.Tensor(arg87_1, 0.999)
        mul_tensor_87: "f32[512]" = torch.ops.aten.mul.Tensor(arg88_1, 0.0010000000000000009);  arg88_1 = None
        add_tensor_43: "f32[512]" = torch.ops.aten.add.Tensor(mul_tensor_86, mul_tensor_87);  mul_tensor_86 = mul_tensor_87 = None
        mul_tensor_88: "f32[512]" = torch.ops.aten.mul.Tensor(arg89_1, 0.999)
        mul_tensor_89: "f32[512]" = torch.ops.aten.mul.Tensor(arg90_1, 0.0010000000000000009);  arg90_1 = None
        add_tensor_44: "f32[512]" = torch.ops.aten.add.Tensor(mul_tensor_88, mul_tensor_89);  mul_tensor_88 = mul_tensor_89 = None
        mul_tensor_90: "f32[128, 512, 1, 1]" = torch.ops.aten.mul.Tensor(arg91_1, 0.999)
        mul_tensor_91: "f32[128, 512, 1, 1]" = torch.ops.aten.mul.Tensor(arg92_1, 0.0010000000000000009);  arg92_1 = None
        add_tensor_45: "f32[128, 512, 1, 1]" = torch.ops.aten.add.Tensor(mul_tensor_90, mul_tensor_91);  mul_tensor_90 = mul_tensor_91 = None
        mul_tensor_92: "f32[128]" = torch.ops.aten.mul.Tensor(arg93_1, 0.999)
        mul_tensor_93: "f32[128]" = torch.ops.aten.mul.Tensor(arg94_1, 0.0010000000000000009);  arg94_1 = None
        add_tensor_46: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_92, mul_tensor_93);  mul_tensor_92 = mul_tensor_93 = None
        mul_tensor_94: "f32[128]" = torch.ops.aten.mul.Tensor(arg95_1, 0.999)
        mul_tensor_95: "f32[128]" = torch.ops.aten.mul.Tensor(arg96_1, 0.0010000000000000009);  arg96_1 = None
        add_tensor_47: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_94, mul_tensor_95);  mul_tensor_94 = mul_tensor_95 = None
        mul_tensor_96: "f32[128, 128, 3, 3]" = torch.ops.aten.mul.Tensor(arg97_1, 0.999)
        mul_tensor_97: "f32[128, 128, 3, 3]" = torch.ops.aten.mul.Tensor(arg98_1, 0.0010000000000000009);  arg98_1 = None
        add_tensor_48: "f32[128, 128, 3, 3]" = torch.ops.aten.add.Tensor(mul_tensor_96, mul_tensor_97);  mul_tensor_96 = mul_tensor_97 = None
        mul_tensor_98: "f32[128]" = torch.ops.aten.mul.Tensor(arg99_1, 0.999)
        mul_tensor_99: "f32[128]" = torch.ops.aten.mul.Tensor(arg100_1, 0.0010000000000000009);  arg100_1 = None
        add_tensor_49: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_98, mul_tensor_99);  mul_tensor_98 = mul_tensor_99 = None
        mul_tensor_100: "f32[128]" = torch.ops.aten.mul.Tensor(arg101_1, 0.999)
        mul_tensor_101: "f32[128]" = torch.ops.aten.mul.Tensor(arg102_1, 0.0010000000000000009);  arg102_1 = None
        add_tensor_50: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_100, mul_tensor_101);  mul_tensor_100 = mul_tensor_101 = None
        mul_tensor_102: "f32[512, 128, 1, 1]" = torch.ops.aten.mul.Tensor(arg103_1, 0.999)
        mul_tensor_103: "f32[512, 128, 1, 1]" = torch.ops.aten.mul.Tensor(arg104_1, 0.0010000000000000009);  arg104_1 = None
        add_tensor_51: "f32[512, 128, 1, 1]" = torch.ops.aten.add.Tensor(mul_tensor_102, mul_tensor_103);  mul_tensor_102 = mul_tensor_103 = None
        mul_tensor_104: "f32[512]" = torch.ops.aten.mul.Tensor(arg105_1, 0.999)
        mul_tensor_105: "f32[512]" = torch.ops.aten.mul.Tensor(arg106_1, 0.0010000000000000009);  arg106_1 = None
        add_tensor_52: "f32[512]" = torch.ops.aten.add.Tensor(mul_tensor_104, mul_tensor_105);  mul_tensor_104 = mul_tensor_105 = None
        mul_tensor_106: "f32[512]" = torch.ops.aten.mul.Tensor(arg107_1, 0.999)
        mul_tensor_107: "f32[512]" = torch.ops.aten.mul.Tensor(arg108_1, 0.0010000000000000009);  arg108_1 = None
        add_tensor_53: "f32[512]" = torch.ops.aten.add.Tensor(mul_tensor_106, mul_tensor_107);  mul_tensor_106 = mul_tensor_107 = None
        mul_tensor_108: "f32[128, 512, 1, 1]" = torch.ops.aten.mul.Tensor(arg109_1, 0.999)
        mul_tensor_109: "f32[128, 512, 1, 1]" = torch.ops.aten.mul.Tensor(arg110_1, 0.0010000000000000009);  arg110_1 = None
        add_tensor_54: "f32[128, 512, 1, 1]" = torch.ops.aten.add.Tensor(mul_tensor_108, mul_tensor_109);  mul_tensor_108 = mul_tensor_109 = None
        mul_tensor_110: "f32[128]" = torch.ops.aten.mul.Tensor(arg111_1, 0.999)
        mul_tensor_111: "f32[128]" = torch.ops.aten.mul.Tensor(arg112_1, 0.0010000000000000009);  arg112_1 = None
        add_tensor_55: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_110, mul_tensor_111);  mul_tensor_110 = mul_tensor_111 = None
        mul_tensor_112: "f32[128]" = torch.ops.aten.mul.Tensor(arg113_1, 0.999)
        mul_tensor_113: "f32[128]" = torch.ops.aten.mul.Tensor(arg114_1, 0.0010000000000000009);  arg114_1 = None
        add_tensor_56: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_112, mul_tensor_113);  mul_tensor_112 = mul_tensor_113 = None
        mul_tensor_114: "f32[128, 128, 3, 3]" = torch.ops.aten.mul.Tensor(arg115_1, 0.999)
        mul_tensor_115: "f32[128, 128, 3, 3]" = torch.ops.aten.mul.Tensor(arg116_1, 0.0010000000000000009);  arg116_1 = None
        add_tensor_57: "f32[128, 128, 3, 3]" = torch.ops.aten.add.Tensor(mul_tensor_114, mul_tensor_115);  mul_tensor_114 = mul_tensor_115 = None
        mul_tensor_116: "f32[128]" = torch.ops.aten.mul.Tensor(arg117_1, 0.999)
        mul_tensor_117: "f32[128]" = torch.ops.aten.mul.Tensor(arg118_1, 0.0010000000000000009);  arg118_1 = None
        add_tensor_58: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_116, mul_tensor_117);  mul_tensor_116 = mul_tensor_117 = None
        mul_tensor_118: "f32[128]" = torch.ops.aten.mul.Tensor(arg119_1, 0.999)
        mul_tensor_119: "f32[128]" = torch.ops.aten.mul.Tensor(arg120_1, 0.0010000000000000009);  arg120_1 = None
        add_tensor_59: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_118, mul_tensor_119);  mul_tensor_118 = mul_tensor_119 = None
        mul_tensor_120: "f32[512, 128, 1, 1]" = torch.ops.aten.mul.Tensor(arg121_1, 0.999)
        mul_tensor_121: "f32[512, 128, 1, 1]" = torch.ops.aten.mul.Tensor(arg122_1, 0.0010000000000000009);  arg122_1 = None
        add_tensor_60: "f32[512, 128, 1, 1]" = torch.ops.aten.add.Tensor(mul_tensor_120, mul_tensor_121);  mul_tensor_120 = mul_tensor_121 = None
        mul_tensor_122: "f32[512]" = torch.ops.aten.mul.Tensor(arg123_1, 0.999)
        mul_tensor_123: "f32[512]" = torch.ops.aten.mul.Tensor(arg124_1, 0.0010000000000000009);  arg124_1 = None
        add_tensor_61: "f32[512]" = torch.ops.aten.add.Tensor(mul_tensor_122, mul_tensor_123);  mul_tensor_122 = mul_tensor_123 = None
        mul_tensor_124: "f32[512]" = torch.ops.aten.mul.Tensor(arg125_1, 0.999)
        mul_tensor_125: "f32[512]" = torch.ops.aten.mul.Tensor(arg126_1, 0.0010000000000000009);  arg126_1 = None
        add_tensor_62: "f32[512]" = torch.ops.aten.add.Tensor(mul_tensor_124, mul_tensor_125);  mul_tensor_124 = mul_tensor_125 = None
        mul_tensor_126: "f32[128, 512, 1, 1]" = torch.ops.aten.mul.Tensor(arg127_1, 0.999)
        mul_tensor_127: "f32[128, 512, 1, 1]" = torch.ops.aten.mul.Tensor(arg128_1, 0.0010000000000000009);  arg128_1 = None
        add_tensor_63: "f32[128, 512, 1, 1]" = torch.ops.aten.add.Tensor(mul_tensor_126, mul_tensor_127);  mul_tensor_126 = mul_tensor_127 = None
        mul_tensor_128: "f32[128]" = torch.ops.aten.mul.Tensor(arg129_1, 0.999)
        mul_tensor_129: "f32[128]" = torch.ops.aten.mul.Tensor(arg130_1, 0.0010000000000000009);  arg130_1 = None
        add_tensor_64: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_128, mul_tensor_129);  mul_tensor_128 = mul_tensor_129 = None
        mul_tensor_130: "f32[128]" = torch.ops.aten.mul.Tensor(arg131_1, 0.999)
        mul_tensor_131: "f32[128]" = torch.ops.aten.mul.Tensor(arg132_1, 0.0010000000000000009);  arg132_1 = None
        add_tensor_65: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_130, mul_tensor_131);  mul_tensor_130 = mul_tensor_131 = None
        mul_tensor_132: "f32[128, 128, 3, 3]" = torch.ops.aten.mul.Tensor(arg133_1, 0.999)
        mul_tensor_133: "f32[128, 128, 3, 3]" = torch.ops.aten.mul.Tensor(arg134_1, 0.0010000000000000009);  arg134_1 = None
        add_tensor_66: "f32[128, 128, 3, 3]" = torch.ops.aten.add.Tensor(mul_tensor_132, mul_tensor_133);  mul_tensor_132 = mul_tensor_133 = None
        mul_tensor_134: "f32[128]" = torch.ops.aten.mul.Tensor(arg135_1, 0.999)
        mul_tensor_135: "f32[128]" = torch.ops.aten.mul.Tensor(arg136_1, 0.0010000000000000009);  arg136_1 = None
        add_tensor_67: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_134, mul_tensor_135);  mul_tensor_134 = mul_tensor_135 = None
        mul_tensor_136: "f32[128]" = torch.ops.aten.mul.Tensor(arg137_1, 0.999)
        mul_tensor_137: "f32[128]" = torch.ops.aten.mul.Tensor(arg138_1, 0.0010000000000000009);  arg138_1 = None
        add_tensor_68: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_136, mul_tensor_137);  mul_tensor_136 = mul_tensor_137 = None
        mul_tensor_138: "f32[512, 128, 1, 1]" = torch.ops.aten.mul.Tensor(arg139_1, 0.999)
        mul_tensor_139: "f32[512, 128, 1, 1]" = torch.ops.aten.mul.Tensor(arg140_1, 0.0010000000000000009);  arg140_1 = None
        add_tensor_69: "f32[512, 128, 1, 1]" = torch.ops.aten.add.Tensor(mul_tensor_138, mul_tensor_139);  mul_tensor_138 = mul_tensor_139 = None
        mul_tensor_140: "f32[512]" = torch.ops.aten.mul.Tensor(arg141_1, 0.999)
        mul_tensor_141: "f32[512]" = torch.ops.aten.mul.Tensor(arg142_1, 0.0010000000000000009);  arg142_1 = None
        add_tensor_70: "f32[512]" = torch.ops.aten.add.Tensor(mul_tensor_140, mul_tensor_141);  mul_tensor_140 = mul_tensor_141 = None
        mul_tensor_142: "f32[512]" = torch.ops.aten.mul.Tensor(arg143_1, 0.999)
        mul_tensor_143: "f32[512]" = torch.ops.aten.mul.Tensor(arg144_1, 0.0010000000000000009);  arg144_1 = None
        add_tensor_71: "f32[512]" = torch.ops.aten.add.Tensor(mul_tensor_142, mul_tensor_143);  mul_tensor_142 = mul_tensor_143 = None
        mul_tensor_144: "f32[256, 512, 1, 1]" = torch.ops.aten.mul.Tensor(arg145_1, 0.999)
        mul_tensor_145: "f32[256, 512, 1, 1]" = torch.ops.aten.mul.Tensor(arg146_1, 0.0010000000000000009);  arg146_1 = None
        add_tensor_72: "f32[256, 512, 1, 1]" = torch.ops.aten.add.Tensor(mul_tensor_144, mul_tensor_145);  mul_tensor_144 = mul_tensor_145 = None
        mul_tensor_146: "f32[256]" = torch.ops.aten.mul.Tensor(arg147_1, 0.999)
        mul_tensor_147: "f32[256]" = torch.ops.aten.mul.Tensor(arg148_1, 0.0010000000000000009);  arg148_1 = None
        add_tensor_73: "f32[256]" = torch.ops.aten.add.Tensor(mul_tensor_146, mul_tensor_147);  mul_tensor_146 = mul_tensor_147 = None
        mul_tensor_148: "f32[256]" = torch.ops.aten.mul.Tensor(arg149_1, 0.999)
        mul_tensor_149: "f32[256]" = torch.ops.aten.mul.Tensor(arg150_1, 0.0010000000000000009);  arg150_1 = None
        add_tensor_74: "f32[256]" = torch.ops.aten.add.Tensor(mul_tensor_148, mul_tensor_149);  mul_tensor_148 = mul_tensor_149 = None
        mul_tensor_150: "f32[256, 256, 3, 3]" = torch.ops.aten.mul.Tensor(arg151_1, 0.999)
        mul_tensor_151: "f32[256, 256, 3, 3]" = torch.ops.aten.mul.Tensor(arg152_1, 0.0010000000000000009);  arg152_1 = None
        add_tensor_75: "f32[256, 256, 3, 3]" = torch.ops.aten.add.Tensor(mul_tensor_150, mul_tensor_151);  mul_tensor_150 = mul_tensor_151 = None
        mul_tensor_152: "f32[256]" = torch.ops.aten.mul.Tensor(arg153_1, 0.999)
        mul_tensor_153: "f32[256]" = torch.ops.aten.mul.Tensor(arg154_1, 0.0010000000000000009);  arg154_1 = None
        add_tensor_76: "f32[256]" = torch.ops.aten.add.Tensor(mul_tensor_152, mul_tensor_153);  mul_tensor_152 = mul_tensor_153 = None
        mul_tensor_154: "f32[256]" = torch.ops.aten.mul.Tensor(arg155_1, 0.999)
        mul_tensor_155: "f32[256]" = torch.ops.aten.mul.Tensor(arg156_1, 0.0010000000000000009);  arg156_1 = None
        add_tensor_77: "f32[256]" = torch.ops.aten.add.Tensor(mul_tensor_154, mul_tensor_155);  mul_tensor_154 = mul_tensor_155 = None
        mul_tensor_156: "f32[1024, 256, 1, 1]" = torch.ops.aten.mul.Tensor(arg157_1, 0.999)
        mul_tensor_157: "f32[1024, 256, 1, 1]" = torch.ops.aten.mul.Tensor(arg158_1, 0.0010000000000000009);  arg158_1 = None
        add_tensor_78: "f32[1024, 256, 1, 1]" = torch.ops.aten.add.Tensor(mul_tensor_156, mul_tensor_157);  mul_tensor_156 = mul_tensor_157 = None
        mul_tensor_158: "f32[1024]" = torch.ops.aten.mul.Tensor(arg159_1, 0.999)
        mul_tensor_159: "f32[1024]" = torch.ops.aten.mul.Tensor(arg160_1, 0.0010000000000000009);  arg160_1 = None
        add_tensor_79: "f32[1024]" = torch.ops.aten.add.Tensor(mul_tensor_158, mul_tensor_159);  mul_tensor_158 = mul_tensor_159 = None
        mul_tensor_160: "f32[1024]" = torch.ops.aten.mul.Tensor(arg161_1, 0.999)
        mul_tensor_161: "f32[1024]" = torch.ops.aten.mul.Tensor(arg162_1, 0.0010000000000000009);  arg162_1 = None
        add_tensor_80: "f32[1024]" = torch.ops.aten.add.Tensor(mul_tensor_160, mul_tensor_161);  mul_tensor_160 = mul_tensor_161 = None
        mul_tensor_162: "f32[1024, 512, 1, 1]" = torch.ops.aten.mul.Tensor(arg163_1, 0.999)
        mul_tensor_163: "f32[1024, 512, 1, 1]" = torch.ops.aten.mul.Tensor(arg164_1, 0.0010000000000000009);  arg164_1 = None
        add_tensor_81: "f32[1024, 512, 1, 1]" = torch.ops.aten.add.Tensor(mul_tensor_162, mul_tensor_163);  mul_tensor_162 = mul_tensor_163 = None
        mul_tensor_164: "f32[1024]" = torch.ops.aten.mul.Tensor(arg165_1, 0.999)
        mul_tensor_165: "f32[1024]" = torch.ops.aten.mul.Tensor(arg166_1, 0.0010000000000000009);  arg166_1 = None
        add_tensor_82: "f32[1024]" = torch.ops.aten.add.Tensor(mul_tensor_164, mul_tensor_165);  mul_tensor_164 = mul_tensor_165 = None
        mul_tensor_166: "f32[1024]" = torch.ops.aten.mul.Tensor(arg167_1, 0.999)
        mul_tensor_167: "f32[1024]" = torch.ops.aten.mul.Tensor(arg168_1, 0.0010000000000000009);  arg168_1 = None
        add_tensor_83: "f32[1024]" = torch.ops.aten.add.Tensor(mul_tensor_166, mul_tensor_167);  mul_tensor_166 = mul_tensor_167 = None
        mul_tensor_168: "f32[256, 1024, 1, 1]" = torch.ops.aten.mul.Tensor(arg169_1, 0.999)
        mul_tensor_169: "f32[256, 1024, 1, 1]" = torch.ops.aten.mul.Tensor(arg170_1, 0.0010000000000000009);  arg170_1 = None
        add_tensor_84: "f32[256, 1024, 1, 1]" = torch.ops.aten.add.Tensor(mul_tensor_168, mul_tensor_169);  mul_tensor_168 = mul_tensor_169 = None
        mul_tensor_170: "f32[256]" = torch.ops.aten.mul.Tensor(arg171_1, 0.999)
        mul_tensor_171: "f32[256]" = torch.ops.aten.mul.Tensor(arg172_1, 0.0010000000000000009);  arg172_1 = None
        add_tensor_85: "f32[256]" = torch.ops.aten.add.Tensor(mul_tensor_170, mul_tensor_171);  mul_tensor_170 = mul_tensor_171 = None
        mul_tensor_172: "f32[256]" = torch.ops.aten.mul.Tensor(arg173_1, 0.999)
        mul_tensor_173: "f32[256]" = torch.ops.aten.mul.Tensor(arg174_1, 0.0010000000000000009);  arg174_1 = None
        add_tensor_86: "f32[256]" = torch.ops.aten.add.Tensor(mul_tensor_172, mul_tensor_173);  mul_tensor_172 = mul_tensor_173 = None
        mul_tensor_174: "f32[256, 256, 3, 3]" = torch.ops.aten.mul.Tensor(arg175_1, 0.999)
        mul_tensor_175: "f32[256, 256, 3, 3]" = torch.ops.aten.mul.Tensor(arg176_1, 0.0010000000000000009);  arg176_1 = None
        add_tensor_87: "f32[256, 256, 3, 3]" = torch.ops.aten.add.Tensor(mul_tensor_174, mul_tensor_175);  mul_tensor_174 = mul_tensor_175 = None
        mul_tensor_176: "f32[256]" = torch.ops.aten.mul.Tensor(arg177_1, 0.999)
        mul_tensor_177: "f32[256]" = torch.ops.aten.mul.Tensor(arg178_1, 0.0010000000000000009);  arg178_1 = None
        add_tensor_88: "f32[256]" = torch.ops.aten.add.Tensor(mul_tensor_176, mul_tensor_177);  mul_tensor_176 = mul_tensor_177 = None
        mul_tensor_178: "f32[256]" = torch.ops.aten.mul.Tensor(arg179_1, 0.999)
        mul_tensor_179: "f32[256]" = torch.ops.aten.mul.Tensor(arg180_1, 0.0010000000000000009);  arg180_1 = None
        add_tensor_89: "f32[256]" = torch.ops.aten.add.Tensor(mul_tensor_178, mul_tensor_179);  mul_tensor_178 = mul_tensor_179 = None
        mul_tensor_180: "f32[1024, 256, 1, 1]" = torch.ops.aten.mul.Tensor(arg181_1, 0.999)
        mul_tensor_181: "f32[1024, 256, 1, 1]" = torch.ops.aten.mul.Tensor(arg182_1, 0.0010000000000000009);  arg182_1 = None
        add_tensor_90: "f32[1024, 256, 1, 1]" = torch.ops.aten.add.Tensor(mul_tensor_180, mul_tensor_181);  mul_tensor_180 = mul_tensor_181 = None
        mul_tensor_182: "f32[1024]" = torch.ops.aten.mul.Tensor(arg183_1, 0.999)
        mul_tensor_183: "f32[1024]" = torch.ops.aten.mul.Tensor(arg184_1, 0.0010000000000000009);  arg184_1 = None
        add_tensor_91: "f32[1024]" = torch.ops.aten.add.Tensor(mul_tensor_182, mul_tensor_183);  mul_tensor_182 = mul_tensor_183 = None
        mul_tensor_184: "f32[1024]" = torch.ops.aten.mul.Tensor(arg185_1, 0.999)
        mul_tensor_185: "f32[1024]" = torch.ops.aten.mul.Tensor(arg186_1, 0.0010000000000000009);  arg186_1 = None
        add_tensor_92: "f32[1024]" = torch.ops.aten.add.Tensor(mul_tensor_184, mul_tensor_185);  mul_tensor_184 = mul_tensor_185 = None
        mul_tensor_186: "f32[256, 1024, 1, 1]" = torch.ops.aten.mul.Tensor(arg187_1, 0.999)
        mul_tensor_187: "f32[256, 1024, 1, 1]" = torch.ops.aten.mul.Tensor(arg188_1, 0.0010000000000000009);  arg188_1 = None
        add_tensor_93: "f32[256, 1024, 1, 1]" = torch.ops.aten.add.Tensor(mul_tensor_186, mul_tensor_187);  mul_tensor_186 = mul_tensor_187 = None
        mul_tensor_188: "f32[256]" = torch.ops.aten.mul.Tensor(arg189_1, 0.999)
        mul_tensor_189: "f32[256]" = torch.ops.aten.mul.Tensor(arg190_1, 0.0010000000000000009);  arg190_1 = None
        add_tensor_94: "f32[256]" = torch.ops.aten.add.Tensor(mul_tensor_188, mul_tensor_189);  mul_tensor_188 = mul_tensor_189 = None
        mul_tensor_190: "f32[256]" = torch.ops.aten.mul.Tensor(arg191_1, 0.999)
        mul_tensor_191: "f32[256]" = torch.ops.aten.mul.Tensor(arg192_1, 0.0010000000000000009);  arg192_1 = None
        add_tensor_95: "f32[256]" = torch.ops.aten.add.Tensor(mul_tensor_190, mul_tensor_191);  mul_tensor_190 = mul_tensor_191 = None
        mul_tensor_192: "f32[256, 256, 3, 3]" = torch.ops.aten.mul.Tensor(arg193_1, 0.999)
        mul_tensor_193: "f32[256, 256, 3, 3]" = torch.ops.aten.mul.Tensor(arg194_1, 0.0010000000000000009);  arg194_1 = None
        add_tensor_96: "f32[256, 256, 3, 3]" = torch.ops.aten.add.Tensor(mul_tensor_192, mul_tensor_193);  mul_tensor_192 = mul_tensor_193 = None
        mul_tensor_194: "f32[256]" = torch.ops.aten.mul.Tensor(arg195_1, 0.999)
        mul_tensor_195: "f32[256]" = torch.ops.aten.mul.Tensor(arg196_1, 0.0010000000000000009);  arg196_1 = None
        add_tensor_97: "f32[256]" = torch.ops.aten.add.Tensor(mul_tensor_194, mul_tensor_195);  mul_tensor_194 = mul_tensor_195 = None
        mul_tensor_196: "f32[256]" = torch.ops.aten.mul.Tensor(arg197_1, 0.999)
        mul_tensor_197: "f32[256]" = torch.ops.aten.mul.Tensor(arg198_1, 0.0010000000000000009);  arg198_1 = None
        add_tensor_98: "f32[256]" = torch.ops.aten.add.Tensor(mul_tensor_196, mul_tensor_197);  mul_tensor_196 = mul_tensor_197 = None
        mul_tensor_198: "f32[1024, 256, 1, 1]" = torch.ops.aten.mul.Tensor(arg199_1, 0.999)
        mul_tensor_199: "f32[1024, 256, 1, 1]" = torch.ops.aten.mul.Tensor(arg200_1, 0.0010000000000000009);  arg200_1 = None
        add_tensor_99: "f32[1024, 256, 1, 1]" = torch.ops.aten.add.Tensor(mul_tensor_198, mul_tensor_199);  mul_tensor_198 = mul_tensor_199 = None
        mul_tensor_200: "f32[1024]" = torch.ops.aten.mul.Tensor(arg201_1, 0.999)
        mul_tensor_201: "f32[1024]" = torch.ops.aten.mul.Tensor(arg202_1, 0.0010000000000000009);  arg202_1 = None
        add_tensor_100: "f32[1024]" = torch.ops.aten.add.Tensor(mul_tensor_200, mul_tensor_201);  mul_tensor_200 = mul_tensor_201 = None
        mul_tensor_202: "f32[1024]" = torch.ops.aten.mul.Tensor(arg203_1, 0.999)
        mul_tensor_203: "f32[1024]" = torch.ops.aten.mul.Tensor(arg204_1, 0.0010000000000000009);  arg204_1 = None
        add_tensor_101: "f32[1024]" = torch.ops.aten.add.Tensor(mul_tensor_202, mul_tensor_203);  mul_tensor_202 = mul_tensor_203 = None
        mul_tensor_204: "f32[256, 1024, 1, 1]" = torch.ops.aten.mul.Tensor(arg205_1, 0.999)
        mul_tensor_205: "f32[256, 1024, 1, 1]" = torch.ops.aten.mul.Tensor(arg206_1, 0.0010000000000000009);  arg206_1 = None
        add_tensor_102: "f32[256, 1024, 1, 1]" = torch.ops.aten.add.Tensor(mul_tensor_204, mul_tensor_205);  mul_tensor_204 = mul_tensor_205 = None
        mul_tensor_206: "f32[256]" = torch.ops.aten.mul.Tensor(arg207_1, 0.999)
        mul_tensor_207: "f32[256]" = torch.ops.aten.mul.Tensor(arg208_1, 0.0010000000000000009);  arg208_1 = None
        add_tensor_103: "f32[256]" = torch.ops.aten.add.Tensor(mul_tensor_206, mul_tensor_207);  mul_tensor_206 = mul_tensor_207 = None
        mul_tensor_208: "f32[256]" = torch.ops.aten.mul.Tensor(arg209_1, 0.999)
        mul_tensor_209: "f32[256]" = torch.ops.aten.mul.Tensor(arg210_1, 0.0010000000000000009);  arg210_1 = None
        add_tensor_104: "f32[256]" = torch.ops.aten.add.Tensor(mul_tensor_208, mul_tensor_209);  mul_tensor_208 = mul_tensor_209 = None
        mul_tensor_210: "f32[256, 256, 3, 3]" = torch.ops.aten.mul.Tensor(arg211_1, 0.999)
        mul_tensor_211: "f32[256, 256, 3, 3]" = torch.ops.aten.mul.Tensor(arg212_1, 0.0010000000000000009);  arg212_1 = None
        add_tensor_105: "f32[256, 256, 3, 3]" = torch.ops.aten.add.Tensor(mul_tensor_210, mul_tensor_211);  mul_tensor_210 = mul_tensor_211 = None
        mul_tensor_212: "f32[256]" = torch.ops.aten.mul.Tensor(arg213_1, 0.999)
        mul_tensor_213: "f32[256]" = torch.ops.aten.mul.Tensor(arg214_1, 0.0010000000000000009);  arg214_1 = None
        add_tensor_106: "f32[256]" = torch.ops.aten.add.Tensor(mul_tensor_212, mul_tensor_213);  mul_tensor_212 = mul_tensor_213 = None
        mul_tensor_214: "f32[256]" = torch.ops.aten.mul.Tensor(arg215_1, 0.999)
        mul_tensor_215: "f32[256]" = torch.ops.aten.mul.Tensor(arg216_1, 0.0010000000000000009);  arg216_1 = None
        add_tensor_107: "f32[256]" = torch.ops.aten.add.Tensor(mul_tensor_214, mul_tensor_215);  mul_tensor_214 = mul_tensor_215 = None
        mul_tensor_216: "f32[1024, 256, 1, 1]" = torch.ops.aten.mul.Tensor(arg217_1, 0.999)
        mul_tensor_217: "f32[1024, 256, 1, 1]" = torch.ops.aten.mul.Tensor(arg218_1, 0.0010000000000000009);  arg218_1 = None
        add_tensor_108: "f32[1024, 256, 1, 1]" = torch.ops.aten.add.Tensor(mul_tensor_216, mul_tensor_217);  mul_tensor_216 = mul_tensor_217 = None
        mul_tensor_218: "f32[1024]" = torch.ops.aten.mul.Tensor(arg219_1, 0.999)
        mul_tensor_219: "f32[1024]" = torch.ops.aten.mul.Tensor(arg220_1, 0.0010000000000000009);  arg220_1 = None
        add_tensor_109: "f32[1024]" = torch.ops.aten.add.Tensor(mul_tensor_218, mul_tensor_219);  mul_tensor_218 = mul_tensor_219 = None
        mul_tensor_220: "f32[1024]" = torch.ops.aten.mul.Tensor(arg221_1, 0.999)
        mul_tensor_221: "f32[1024]" = torch.ops.aten.mul.Tensor(arg222_1, 0.0010000000000000009);  arg222_1 = None
        add_tensor_110: "f32[1024]" = torch.ops.aten.add.Tensor(mul_tensor_220, mul_tensor_221);  mul_tensor_220 = mul_tensor_221 = None
        mul_tensor_222: "f32[256, 1024, 1, 1]" = torch.ops.aten.mul.Tensor(arg223_1, 0.999)
        mul_tensor_223: "f32[256, 1024, 1, 1]" = torch.ops.aten.mul.Tensor(arg224_1, 0.0010000000000000009);  arg224_1 = None
        add_tensor_111: "f32[256, 1024, 1, 1]" = torch.ops.aten.add.Tensor(mul_tensor_222, mul_tensor_223);  mul_tensor_222 = mul_tensor_223 = None
        mul_tensor_224: "f32[256]" = torch.ops.aten.mul.Tensor(arg225_1, 0.999)
        mul_tensor_225: "f32[256]" = torch.ops.aten.mul.Tensor(arg226_1, 0.0010000000000000009);  arg226_1 = None
        add_tensor_112: "f32[256]" = torch.ops.aten.add.Tensor(mul_tensor_224, mul_tensor_225);  mul_tensor_224 = mul_tensor_225 = None
        mul_tensor_226: "f32[256]" = torch.ops.aten.mul.Tensor(arg227_1, 0.999)
        mul_tensor_227: "f32[256]" = torch.ops.aten.mul.Tensor(arg228_1, 0.0010000000000000009);  arg228_1 = None
        add_tensor_113: "f32[256]" = torch.ops.aten.add.Tensor(mul_tensor_226, mul_tensor_227);  mul_tensor_226 = mul_tensor_227 = None
        mul_tensor_228: "f32[256, 256, 3, 3]" = torch.ops.aten.mul.Tensor(arg229_1, 0.999)
        mul_tensor_229: "f32[256, 256, 3, 3]" = torch.ops.aten.mul.Tensor(arg230_1, 0.0010000000000000009);  arg230_1 = None
        add_tensor_114: "f32[256, 256, 3, 3]" = torch.ops.aten.add.Tensor(mul_tensor_228, mul_tensor_229);  mul_tensor_228 = mul_tensor_229 = None
        mul_tensor_230: "f32[256]" = torch.ops.aten.mul.Tensor(arg231_1, 0.999)
        mul_tensor_231: "f32[256]" = torch.ops.aten.mul.Tensor(arg232_1, 0.0010000000000000009);  arg232_1 = None
        add_tensor_115: "f32[256]" = torch.ops.aten.add.Tensor(mul_tensor_230, mul_tensor_231);  mul_tensor_230 = mul_tensor_231 = None
        mul_tensor_232: "f32[256]" = torch.ops.aten.mul.Tensor(arg233_1, 0.999)
        mul_tensor_233: "f32[256]" = torch.ops.aten.mul.Tensor(arg234_1, 0.0010000000000000009);  arg234_1 = None
        add_tensor_116: "f32[256]" = torch.ops.aten.add.Tensor(mul_tensor_232, mul_tensor_233);  mul_tensor_232 = mul_tensor_233 = None
        mul_tensor_234: "f32[1024, 256, 1, 1]" = torch.ops.aten.mul.Tensor(arg235_1, 0.999)
        mul_tensor_235: "f32[1024, 256, 1, 1]" = torch.ops.aten.mul.Tensor(arg236_1, 0.0010000000000000009);  arg236_1 = None
        add_tensor_117: "f32[1024, 256, 1, 1]" = torch.ops.aten.add.Tensor(mul_tensor_234, mul_tensor_235);  mul_tensor_234 = mul_tensor_235 = None
        mul_tensor_236: "f32[1024]" = torch.ops.aten.mul.Tensor(arg237_1, 0.999)
        mul_tensor_237: "f32[1024]" = torch.ops.aten.mul.Tensor(arg238_1, 0.0010000000000000009);  arg238_1 = None
        add_tensor_118: "f32[1024]" = torch.ops.aten.add.Tensor(mul_tensor_236, mul_tensor_237);  mul_tensor_236 = mul_tensor_237 = None
        mul_tensor_238: "f32[1024]" = torch.ops.aten.mul.Tensor(arg239_1, 0.999)
        mul_tensor_239: "f32[1024]" = torch.ops.aten.mul.Tensor(arg240_1, 0.0010000000000000009);  arg240_1 = None
        add_tensor_119: "f32[1024]" = torch.ops.aten.add.Tensor(mul_tensor_238, mul_tensor_239);  mul_tensor_238 = mul_tensor_239 = None
        mul_tensor_240: "f32[256, 1024, 1, 1]" = torch.ops.aten.mul.Tensor(arg241_1, 0.999)
        mul_tensor_241: "f32[256, 1024, 1, 1]" = torch.ops.aten.mul.Tensor(arg242_1, 0.0010000000000000009);  arg242_1 = None
        add_tensor_120: "f32[256, 1024, 1, 1]" = torch.ops.aten.add.Tensor(mul_tensor_240, mul_tensor_241);  mul_tensor_240 = mul_tensor_241 = None
        mul_tensor_242: "f32[256]" = torch.ops.aten.mul.Tensor(arg243_1, 0.999)
        mul_tensor_243: "f32[256]" = torch.ops.aten.mul.Tensor(arg244_1, 0.0010000000000000009);  arg244_1 = None
        add_tensor_121: "f32[256]" = torch.ops.aten.add.Tensor(mul_tensor_242, mul_tensor_243);  mul_tensor_242 = mul_tensor_243 = None
        mul_tensor_244: "f32[256]" = torch.ops.aten.mul.Tensor(arg245_1, 0.999)
        mul_tensor_245: "f32[256]" = torch.ops.aten.mul.Tensor(arg246_1, 0.0010000000000000009);  arg246_1 = None
        add_tensor_122: "f32[256]" = torch.ops.aten.add.Tensor(mul_tensor_244, mul_tensor_245);  mul_tensor_244 = mul_tensor_245 = None
        mul_tensor_246: "f32[256, 256, 3, 3]" = torch.ops.aten.mul.Tensor(arg247_1, 0.999)
        mul_tensor_247: "f32[256, 256, 3, 3]" = torch.ops.aten.mul.Tensor(arg248_1, 0.0010000000000000009);  arg248_1 = None
        add_tensor_123: "f32[256, 256, 3, 3]" = torch.ops.aten.add.Tensor(mul_tensor_246, mul_tensor_247);  mul_tensor_246 = mul_tensor_247 = None
        mul_tensor_248: "f32[256]" = torch.ops.aten.mul.Tensor(arg249_1, 0.999)
        mul_tensor_249: "f32[256]" = torch.ops.aten.mul.Tensor(arg250_1, 0.0010000000000000009);  arg250_1 = None
        add_tensor_124: "f32[256]" = torch.ops.aten.add.Tensor(mul_tensor_248, mul_tensor_249);  mul_tensor_248 = mul_tensor_249 = None
        mul_tensor_250: "f32[256]" = torch.ops.aten.mul.Tensor(arg251_1, 0.999)
        mul_tensor_251: "f32[256]" = torch.ops.aten.mul.Tensor(arg252_1, 0.0010000000000000009);  arg252_1 = None
        add_tensor_125: "f32[256]" = torch.ops.aten.add.Tensor(mul_tensor_250, mul_tensor_251);  mul_tensor_250 = mul_tensor_251 = None
        mul_tensor_252: "f32[1024, 256, 1, 1]" = torch.ops.aten.mul.Tensor(arg253_1, 0.999)
        mul_tensor_253: "f32[1024, 256, 1, 1]" = torch.ops.aten.mul.Tensor(arg254_1, 0.0010000000000000009);  arg254_1 = None
        add_tensor_126: "f32[1024, 256, 1, 1]" = torch.ops.aten.add.Tensor(mul_tensor_252, mul_tensor_253);  mul_tensor_252 = mul_tensor_253 = None
        mul_tensor_254: "f32[1024]" = torch.ops.aten.mul.Tensor(arg255_1, 0.999)
        mul_tensor_255: "f32[1024]" = torch.ops.aten.mul.Tensor(arg256_1, 0.0010000000000000009);  arg256_1 = None
        add_tensor_127: "f32[1024]" = torch.ops.aten.add.Tensor(mul_tensor_254, mul_tensor_255);  mul_tensor_254 = mul_tensor_255 = None
        mul_tensor_256: "f32[1024]" = torch.ops.aten.mul.Tensor(arg257_1, 0.999)
        mul_tensor_257: "f32[1024]" = torch.ops.aten.mul.Tensor(arg258_1, 0.0010000000000000009);  arg258_1 = None
        add_tensor_128: "f32[1024]" = torch.ops.aten.add.Tensor(mul_tensor_256, mul_tensor_257);  mul_tensor_256 = mul_tensor_257 = None
        mul_tensor_258: "f32[512, 1024, 1, 1]" = torch.ops.aten.mul.Tensor(arg259_1, 0.999)
        mul_tensor_259: "f32[512, 1024, 1, 1]" = torch.ops.aten.mul.Tensor(arg260_1, 0.0010000000000000009);  arg260_1 = None
        add_tensor_129: "f32[512, 1024, 1, 1]" = torch.ops.aten.add.Tensor(mul_tensor_258, mul_tensor_259);  mul_tensor_258 = mul_tensor_259 = None
        mul_tensor_260: "f32[512]" = torch.ops.aten.mul.Tensor(arg261_1, 0.999)
        mul_tensor_261: "f32[512]" = torch.ops.aten.mul.Tensor(arg262_1, 0.0010000000000000009);  arg262_1 = None
        add_tensor_130: "f32[512]" = torch.ops.aten.add.Tensor(mul_tensor_260, mul_tensor_261);  mul_tensor_260 = mul_tensor_261 = None
        mul_tensor_262: "f32[512]" = torch.ops.aten.mul.Tensor(arg263_1, 0.999)
        mul_tensor_263: "f32[512]" = torch.ops.aten.mul.Tensor(arg264_1, 0.0010000000000000009);  arg264_1 = None
        add_tensor_131: "f32[512]" = torch.ops.aten.add.Tensor(mul_tensor_262, mul_tensor_263);  mul_tensor_262 = mul_tensor_263 = None
        mul_tensor_264: "f32[512, 512, 3, 3]" = torch.ops.aten.mul.Tensor(arg265_1, 0.999)
        mul_tensor_265: "f32[512, 512, 3, 3]" = torch.ops.aten.mul.Tensor(arg266_1, 0.0010000000000000009);  arg266_1 = None
        add_tensor_132: "f32[512, 512, 3, 3]" = torch.ops.aten.add.Tensor(mul_tensor_264, mul_tensor_265);  mul_tensor_264 = mul_tensor_265 = None
        mul_tensor_266: "f32[512]" = torch.ops.aten.mul.Tensor(arg267_1, 0.999)
        mul_tensor_267: "f32[512]" = torch.ops.aten.mul.Tensor(arg268_1, 0.0010000000000000009);  arg268_1 = None
        add_tensor_133: "f32[512]" = torch.ops.aten.add.Tensor(mul_tensor_266, mul_tensor_267);  mul_tensor_266 = mul_tensor_267 = None
        mul_tensor_268: "f32[512]" = torch.ops.aten.mul.Tensor(arg269_1, 0.999)
        mul_tensor_269: "f32[512]" = torch.ops.aten.mul.Tensor(arg270_1, 0.0010000000000000009);  arg270_1 = None
        add_tensor_134: "f32[512]" = torch.ops.aten.add.Tensor(mul_tensor_268, mul_tensor_269);  mul_tensor_268 = mul_tensor_269 = None
        mul_tensor_270: "f32[2048, 512, 1, 1]" = torch.ops.aten.mul.Tensor(arg271_1, 0.999)
        mul_tensor_271: "f32[2048, 512, 1, 1]" = torch.ops.aten.mul.Tensor(arg272_1, 0.0010000000000000009);  arg272_1 = None
        add_tensor_135: "f32[2048, 512, 1, 1]" = torch.ops.aten.add.Tensor(mul_tensor_270, mul_tensor_271);  mul_tensor_270 = mul_tensor_271 = None
        mul_tensor_272: "f32[2048]" = torch.ops.aten.mul.Tensor(arg273_1, 0.999)
        mul_tensor_273: "f32[2048]" = torch.ops.aten.mul.Tensor(arg274_1, 0.0010000000000000009);  arg274_1 = None
        add_tensor_136: "f32[2048]" = torch.ops.aten.add.Tensor(mul_tensor_272, mul_tensor_273);  mul_tensor_272 = mul_tensor_273 = None
        mul_tensor_274: "f32[2048]" = torch.ops.aten.mul.Tensor(arg275_1, 0.999)
        mul_tensor_275: "f32[2048]" = torch.ops.aten.mul.Tensor(arg276_1, 0.0010000000000000009);  arg276_1 = None
        add_tensor_137: "f32[2048]" = torch.ops.aten.add.Tensor(mul_tensor_274, mul_tensor_275);  mul_tensor_274 = mul_tensor_275 = None
        mul_tensor_276: "f32[2048, 1024, 1, 1]" = torch.ops.aten.mul.Tensor(arg277_1, 0.999)
        mul_tensor_277: "f32[2048, 1024, 1, 1]" = torch.ops.aten.mul.Tensor(arg278_1, 0.0010000000000000009);  arg278_1 = None
        add_tensor_138: "f32[2048, 1024, 1, 1]" = torch.ops.aten.add.Tensor(mul_tensor_276, mul_tensor_277);  mul_tensor_276 = mul_tensor_277 = None
        mul_tensor_278: "f32[2048]" = torch.ops.aten.mul.Tensor(arg279_1, 0.999)
        mul_tensor_279: "f32[2048]" = torch.ops.aten.mul.Tensor(arg280_1, 0.0010000000000000009);  arg280_1 = None
        add_tensor_139: "f32[2048]" = torch.ops.aten.add.Tensor(mul_tensor_278, mul_tensor_279);  mul_tensor_278 = mul_tensor_279 = None
        mul_tensor_280: "f32[2048]" = torch.ops.aten.mul.Tensor(arg281_1, 0.999)
        mul_tensor_281: "f32[2048]" = torch.ops.aten.mul.Tensor(arg282_1, 0.0010000000000000009);  arg282_1 = None
        add_tensor_140: "f32[2048]" = torch.ops.aten.add.Tensor(mul_tensor_280, mul_tensor_281);  mul_tensor_280 = mul_tensor_281 = None
        mul_tensor_282: "f32[512, 2048, 1, 1]" = torch.ops.aten.mul.Tensor(arg283_1, 0.999)
        mul_tensor_283: "f32[512, 2048, 1, 1]" = torch.ops.aten.mul.Tensor(arg284_1, 0.0010000000000000009);  arg284_1 = None
        add_tensor_141: "f32[512, 2048, 1, 1]" = torch.ops.aten.add.Tensor(mul_tensor_282, mul_tensor_283);  mul_tensor_282 = mul_tensor_283 = None
        mul_tensor_284: "f32[512]" = torch.ops.aten.mul.Tensor(arg285_1, 0.999)
        mul_tensor_285: "f32[512]" = torch.ops.aten.mul.Tensor(arg286_1, 0.0010000000000000009);  arg286_1 = None
        add_tensor_142: "f32[512]" = torch.ops.aten.add.Tensor(mul_tensor_284, mul_tensor_285);  mul_tensor_284 = mul_tensor_285 = None
        mul_tensor_286: "f32[512]" = torch.ops.aten.mul.Tensor(arg287_1, 0.999)
        mul_tensor_287: "f32[512]" = torch.ops.aten.mul.Tensor(arg288_1, 0.0010000000000000009);  arg288_1 = None
        add_tensor_143: "f32[512]" = torch.ops.aten.add.Tensor(mul_tensor_286, mul_tensor_287);  mul_tensor_286 = mul_tensor_287 = None
        mul_tensor_288: "f32[512, 512, 3, 3]" = torch.ops.aten.mul.Tensor(arg289_1, 0.999)
        mul_tensor_289: "f32[512, 512, 3, 3]" = torch.ops.aten.mul.Tensor(arg290_1, 0.0010000000000000009);  arg290_1 = None
        add_tensor_144: "f32[512, 512, 3, 3]" = torch.ops.aten.add.Tensor(mul_tensor_288, mul_tensor_289);  mul_tensor_288 = mul_tensor_289 = None
        mul_tensor_290: "f32[512]" = torch.ops.aten.mul.Tensor(arg291_1, 0.999)
        mul_tensor_291: "f32[512]" = torch.ops.aten.mul.Tensor(arg292_1, 0.0010000000000000009);  arg292_1 = None
        add_tensor_145: "f32[512]" = torch.ops.aten.add.Tensor(mul_tensor_290, mul_tensor_291);  mul_tensor_290 = mul_tensor_291 = None
        mul_tensor_292: "f32[512]" = torch.ops.aten.mul.Tensor(arg293_1, 0.999)
        mul_tensor_293: "f32[512]" = torch.ops.aten.mul.Tensor(arg294_1, 0.0010000000000000009);  arg294_1 = None
        add_tensor_146: "f32[512]" = torch.ops.aten.add.Tensor(mul_tensor_292, mul_tensor_293);  mul_tensor_292 = mul_tensor_293 = None
        mul_tensor_294: "f32[2048, 512, 1, 1]" = torch.ops.aten.mul.Tensor(arg295_1, 0.999)
        mul_tensor_295: "f32[2048, 512, 1, 1]" = torch.ops.aten.mul.Tensor(arg296_1, 0.0010000000000000009);  arg296_1 = None
        add_tensor_147: "f32[2048, 512, 1, 1]" = torch.ops.aten.add.Tensor(mul_tensor_294, mul_tensor_295);  mul_tensor_294 = mul_tensor_295 = None
        mul_tensor_296: "f32[2048]" = torch.ops.aten.mul.Tensor(arg297_1, 0.999)
        mul_tensor_297: "f32[2048]" = torch.ops.aten.mul.Tensor(arg298_1, 0.0010000000000000009);  arg298_1 = None
        add_tensor_148: "f32[2048]" = torch.ops.aten.add.Tensor(mul_tensor_296, mul_tensor_297);  mul_tensor_296 = mul_tensor_297 = None
        mul_tensor_298: "f32[2048]" = torch.ops.aten.mul.Tensor(arg299_1, 0.999)
        mul_tensor_299: "f32[2048]" = torch.ops.aten.mul.Tensor(arg300_1, 0.0010000000000000009);  arg300_1 = None
        add_tensor_149: "f32[2048]" = torch.ops.aten.add.Tensor(mul_tensor_298, mul_tensor_299);  mul_tensor_298 = mul_tensor_299 = None
        mul_tensor_300: "f32[512, 2048, 1, 1]" = torch.ops.aten.mul.Tensor(arg301_1, 0.999)
        mul_tensor_301: "f32[512, 2048, 1, 1]" = torch.ops.aten.mul.Tensor(arg302_1, 0.0010000000000000009);  arg302_1 = None
        add_tensor_150: "f32[512, 2048, 1, 1]" = torch.ops.aten.add.Tensor(mul_tensor_300, mul_tensor_301);  mul_tensor_300 = mul_tensor_301 = None
        mul_tensor_302: "f32[512]" = torch.ops.aten.mul.Tensor(arg303_1, 0.999)
        mul_tensor_303: "f32[512]" = torch.ops.aten.mul.Tensor(arg304_1, 0.0010000000000000009);  arg304_1 = None
        add_tensor_151: "f32[512]" = torch.ops.aten.add.Tensor(mul_tensor_302, mul_tensor_303);  mul_tensor_302 = mul_tensor_303 = None
        mul_tensor_304: "f32[512]" = torch.ops.aten.mul.Tensor(arg305_1, 0.999)
        mul_tensor_305: "f32[512]" = torch.ops.aten.mul.Tensor(arg306_1, 0.0010000000000000009);  arg306_1 = None
        add_tensor_152: "f32[512]" = torch.ops.aten.add.Tensor(mul_tensor_304, mul_tensor_305);  mul_tensor_304 = mul_tensor_305 = None
        mul_tensor_306: "f32[512, 512, 3, 3]" = torch.ops.aten.mul.Tensor(arg307_1, 0.999)
        mul_tensor_307: "f32[512, 512, 3, 3]" = torch.ops.aten.mul.Tensor(arg308_1, 0.0010000000000000009);  arg308_1 = None
        add_tensor_153: "f32[512, 512, 3, 3]" = torch.ops.aten.add.Tensor(mul_tensor_306, mul_tensor_307);  mul_tensor_306 = mul_tensor_307 = None
        mul_tensor_308: "f32[512]" = torch.ops.aten.mul.Tensor(arg309_1, 0.999)
        mul_tensor_309: "f32[512]" = torch.ops.aten.mul.Tensor(arg310_1, 0.0010000000000000009);  arg310_1 = None
        add_tensor_154: "f32[512]" = torch.ops.aten.add.Tensor(mul_tensor_308, mul_tensor_309);  mul_tensor_308 = mul_tensor_309 = None
        mul_tensor_310: "f32[512]" = torch.ops.aten.mul.Tensor(arg311_1, 0.999)
        mul_tensor_311: "f32[512]" = torch.ops.aten.mul.Tensor(arg312_1, 0.0010000000000000009);  arg312_1 = None
        add_tensor_155: "f32[512]" = torch.ops.aten.add.Tensor(mul_tensor_310, mul_tensor_311);  mul_tensor_310 = mul_tensor_311 = None
        mul_tensor_312: "f32[2048, 512, 1, 1]" = torch.ops.aten.mul.Tensor(arg313_1, 0.999)
        mul_tensor_313: "f32[2048, 512, 1, 1]" = torch.ops.aten.mul.Tensor(arg314_1, 0.0010000000000000009);  arg314_1 = None
        add_tensor_156: "f32[2048, 512, 1, 1]" = torch.ops.aten.add.Tensor(mul_tensor_312, mul_tensor_313);  mul_tensor_312 = mul_tensor_313 = None
        mul_tensor_314: "f32[2048]" = torch.ops.aten.mul.Tensor(arg315_1, 0.999)
        mul_tensor_315: "f32[2048]" = torch.ops.aten.mul.Tensor(arg316_1, 0.0010000000000000009);  arg316_1 = None
        add_tensor_157: "f32[2048]" = torch.ops.aten.add.Tensor(mul_tensor_314, mul_tensor_315);  mul_tensor_314 = mul_tensor_315 = None
        mul_tensor_316: "f32[2048]" = torch.ops.aten.mul.Tensor(arg317_1, 0.999)
        mul_tensor_317: "f32[2048]" = torch.ops.aten.mul.Tensor(arg318_1, 0.0010000000000000009);  arg318_1 = None
        add_tensor_158: "f32[2048]" = torch.ops.aten.add.Tensor(mul_tensor_316, mul_tensor_317);  mul_tensor_316 = mul_tensor_317 = None
        mul_tensor_318: "f32[128, 2048]" = torch.ops.aten.mul.Tensor(arg319_1, 0.999)
        mul_tensor_319: "f32[128, 2048]" = torch.ops.aten.mul.Tensor(arg320_1, 0.0010000000000000009);  arg320_1 = None
        add_tensor_159: "f32[128, 2048]" = torch.ops.aten.add.Tensor(mul_tensor_318, mul_tensor_319);  mul_tensor_318 = mul_tensor_319 = None
        mul_tensor_320: "f32[128]" = torch.ops.aten.mul.Tensor(arg321_1, 0.999)
        mul_tensor_321: "f32[128]" = torch.ops.aten.mul.Tensor(arg322_1, 0.0010000000000000009);  arg322_1 = None
        add_tensor_160: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_320, mul_tensor_321);  mul_tensor_320 = mul_tensor_321 = None
        copy__default: "f32[64, 3, 7, 7]" = torch.ops.aten.copy_.default(arg1_1, add_tensor);  arg1_1 = add_tensor = None
        copy__default_1: "f32[64]" = torch.ops.aten.copy_.default(arg3_1, add_tensor_1);  arg3_1 = add_tensor_1 = None
        copy__default_2: "f32[64]" = torch.ops.aten.copy_.default(arg5_1, add_tensor_2);  arg5_1 = add_tensor_2 = None
        copy__default_3: "f32[64, 64, 1, 1]" = torch.ops.aten.copy_.default(arg7_1, add_tensor_3);  arg7_1 = add_tensor_3 = None
        copy__default_4: "f32[64]" = torch.ops.aten.copy_.default(arg9_1, add_tensor_4);  arg9_1 = add_tensor_4 = None
        copy__default_5: "f32[64]" = torch.ops.aten.copy_.default(arg11_1, add_tensor_5);  arg11_1 = add_tensor_5 = None
        copy__default_6: "f32[64, 64, 3, 3]" = torch.ops.aten.copy_.default(arg13_1, add_tensor_6);  arg13_1 = add_tensor_6 = None
        copy__default_7: "f32[64]" = torch.ops.aten.copy_.default(arg15_1, add_tensor_7);  arg15_1 = add_tensor_7 = None
        copy__default_8: "f32[64]" = torch.ops.aten.copy_.default(arg17_1, add_tensor_8);  arg17_1 = add_tensor_8 = None
        copy__default_9: "f32[256, 64, 1, 1]" = torch.ops.aten.copy_.default(arg19_1, add_tensor_9);  arg19_1 = add_tensor_9 = None
        copy__default_10: "f32[256]" = torch.ops.aten.copy_.default(arg21_1, add_tensor_10);  arg21_1 = add_tensor_10 = None
        copy__default_11: "f32[256]" = torch.ops.aten.copy_.default(arg23_1, add_tensor_11);  arg23_1 = add_tensor_11 = None
        copy__default_12: "f32[256, 64, 1, 1]" = torch.ops.aten.copy_.default(arg25_1, add_tensor_12);  arg25_1 = add_tensor_12 = None
        copy__default_13: "f32[256]" = torch.ops.aten.copy_.default(arg27_1, add_tensor_13);  arg27_1 = add_tensor_13 = None
        copy__default_14: "f32[256]" = torch.ops.aten.copy_.default(arg29_1, add_tensor_14);  arg29_1 = add_tensor_14 = None
        copy__default_15: "f32[64, 256, 1, 1]" = torch.ops.aten.copy_.default(arg31_1, add_tensor_15);  arg31_1 = add_tensor_15 = None
        copy__default_16: "f32[64]" = torch.ops.aten.copy_.default(arg33_1, add_tensor_16);  arg33_1 = add_tensor_16 = None
        copy__default_17: "f32[64]" = torch.ops.aten.copy_.default(arg35_1, add_tensor_17);  arg35_1 = add_tensor_17 = None
        copy__default_18: "f32[64, 64, 3, 3]" = torch.ops.aten.copy_.default(arg37_1, add_tensor_18);  arg37_1 = add_tensor_18 = None
        copy__default_19: "f32[64]" = torch.ops.aten.copy_.default(arg39_1, add_tensor_19);  arg39_1 = add_tensor_19 = None
        copy__default_20: "f32[64]" = torch.ops.aten.copy_.default(arg41_1, add_tensor_20);  arg41_1 = add_tensor_20 = None
        copy__default_21: "f32[256, 64, 1, 1]" = torch.ops.aten.copy_.default(arg43_1, add_tensor_21);  arg43_1 = add_tensor_21 = None
        copy__default_22: "f32[256]" = torch.ops.aten.copy_.default(arg45_1, add_tensor_22);  arg45_1 = add_tensor_22 = None
        copy__default_23: "f32[256]" = torch.ops.aten.copy_.default(arg47_1, add_tensor_23);  arg47_1 = add_tensor_23 = None
        copy__default_24: "f32[64, 256, 1, 1]" = torch.ops.aten.copy_.default(arg49_1, add_tensor_24);  arg49_1 = add_tensor_24 = None
        copy__default_25: "f32[64]" = torch.ops.aten.copy_.default(arg51_1, add_tensor_25);  arg51_1 = add_tensor_25 = None
        copy__default_26: "f32[64]" = torch.ops.aten.copy_.default(arg53_1, add_tensor_26);  arg53_1 = add_tensor_26 = None
        copy__default_27: "f32[64, 64, 3, 3]" = torch.ops.aten.copy_.default(arg55_1, add_tensor_27);  arg55_1 = add_tensor_27 = None
        copy__default_28: "f32[64]" = torch.ops.aten.copy_.default(arg57_1, add_tensor_28);  arg57_1 = add_tensor_28 = None
        copy__default_29: "f32[64]" = torch.ops.aten.copy_.default(arg59_1, add_tensor_29);  arg59_1 = add_tensor_29 = None
        copy__default_30: "f32[256, 64, 1, 1]" = torch.ops.aten.copy_.default(arg61_1, add_tensor_30);  arg61_1 = add_tensor_30 = None
        copy__default_31: "f32[256]" = torch.ops.aten.copy_.default(arg63_1, add_tensor_31);  arg63_1 = add_tensor_31 = None
        copy__default_32: "f32[256]" = torch.ops.aten.copy_.default(arg65_1, add_tensor_32);  arg65_1 = add_tensor_32 = None
        copy__default_33: "f32[128, 256, 1, 1]" = torch.ops.aten.copy_.default(arg67_1, add_tensor_33);  arg67_1 = add_tensor_33 = None
        copy__default_34: "f32[128]" = torch.ops.aten.copy_.default(arg69_1, add_tensor_34);  arg69_1 = add_tensor_34 = None
        copy__default_35: "f32[128]" = torch.ops.aten.copy_.default(arg71_1, add_tensor_35);  arg71_1 = add_tensor_35 = None
        copy__default_36: "f32[128, 128, 3, 3]" = torch.ops.aten.copy_.default(arg73_1, add_tensor_36);  arg73_1 = add_tensor_36 = None
        copy__default_37: "f32[128]" = torch.ops.aten.copy_.default(arg75_1, add_tensor_37);  arg75_1 = add_tensor_37 = None
        copy__default_38: "f32[128]" = torch.ops.aten.copy_.default(arg77_1, add_tensor_38);  arg77_1 = add_tensor_38 = None
        copy__default_39: "f32[512, 128, 1, 1]" = torch.ops.aten.copy_.default(arg79_1, add_tensor_39);  arg79_1 = add_tensor_39 = None
        copy__default_40: "f32[512]" = torch.ops.aten.copy_.default(arg81_1, add_tensor_40);  arg81_1 = add_tensor_40 = None
        copy__default_41: "f32[512]" = torch.ops.aten.copy_.default(arg83_1, add_tensor_41);  arg83_1 = add_tensor_41 = None
        copy__default_42: "f32[512, 256, 1, 1]" = torch.ops.aten.copy_.default(arg85_1, add_tensor_42);  arg85_1 = add_tensor_42 = None
        copy__default_43: "f32[512]" = torch.ops.aten.copy_.default(arg87_1, add_tensor_43);  arg87_1 = add_tensor_43 = None
        copy__default_44: "f32[512]" = torch.ops.aten.copy_.default(arg89_1, add_tensor_44);  arg89_1 = add_tensor_44 = None
        copy__default_45: "f32[128, 512, 1, 1]" = torch.ops.aten.copy_.default(arg91_1, add_tensor_45);  arg91_1 = add_tensor_45 = None
        copy__default_46: "f32[128]" = torch.ops.aten.copy_.default(arg93_1, add_tensor_46);  arg93_1 = add_tensor_46 = None
        copy__default_47: "f32[128]" = torch.ops.aten.copy_.default(arg95_1, add_tensor_47);  arg95_1 = add_tensor_47 = None
        copy__default_48: "f32[128, 128, 3, 3]" = torch.ops.aten.copy_.default(arg97_1, add_tensor_48);  arg97_1 = add_tensor_48 = None
        copy__default_49: "f32[128]" = torch.ops.aten.copy_.default(arg99_1, add_tensor_49);  arg99_1 = add_tensor_49 = None
        copy__default_50: "f32[128]" = torch.ops.aten.copy_.default(arg101_1, add_tensor_50);  arg101_1 = add_tensor_50 = None
        copy__default_51: "f32[512, 128, 1, 1]" = torch.ops.aten.copy_.default(arg103_1, add_tensor_51);  arg103_1 = add_tensor_51 = None
        copy__default_52: "f32[512]" = torch.ops.aten.copy_.default(arg105_1, add_tensor_52);  arg105_1 = add_tensor_52 = None
        copy__default_53: "f32[512]" = torch.ops.aten.copy_.default(arg107_1, add_tensor_53);  arg107_1 = add_tensor_53 = None
        copy__default_54: "f32[128, 512, 1, 1]" = torch.ops.aten.copy_.default(arg109_1, add_tensor_54);  arg109_1 = add_tensor_54 = None
        copy__default_55: "f32[128]" = torch.ops.aten.copy_.default(arg111_1, add_tensor_55);  arg111_1 = add_tensor_55 = None
        copy__default_56: "f32[128]" = torch.ops.aten.copy_.default(arg113_1, add_tensor_56);  arg113_1 = add_tensor_56 = None
        copy__default_57: "f32[128, 128, 3, 3]" = torch.ops.aten.copy_.default(arg115_1, add_tensor_57);  arg115_1 = add_tensor_57 = None
        copy__default_58: "f32[128]" = torch.ops.aten.copy_.default(arg117_1, add_tensor_58);  arg117_1 = add_tensor_58 = None
        copy__default_59: "f32[128]" = torch.ops.aten.copy_.default(arg119_1, add_tensor_59);  arg119_1 = add_tensor_59 = None
        copy__default_60: "f32[512, 128, 1, 1]" = torch.ops.aten.copy_.default(arg121_1, add_tensor_60);  arg121_1 = add_tensor_60 = None
        copy__default_61: "f32[512]" = torch.ops.aten.copy_.default(arg123_1, add_tensor_61);  arg123_1 = add_tensor_61 = None
        copy__default_62: "f32[512]" = torch.ops.aten.copy_.default(arg125_1, add_tensor_62);  arg125_1 = add_tensor_62 = None
        copy__default_63: "f32[128, 512, 1, 1]" = torch.ops.aten.copy_.default(arg127_1, add_tensor_63);  arg127_1 = add_tensor_63 = None
        copy__default_64: "f32[128]" = torch.ops.aten.copy_.default(arg129_1, add_tensor_64);  arg129_1 = add_tensor_64 = None
        copy__default_65: "f32[128]" = torch.ops.aten.copy_.default(arg131_1, add_tensor_65);  arg131_1 = add_tensor_65 = None
        copy__default_66: "f32[128, 128, 3, 3]" = torch.ops.aten.copy_.default(arg133_1, add_tensor_66);  arg133_1 = add_tensor_66 = None
        copy__default_67: "f32[128]" = torch.ops.aten.copy_.default(arg135_1, add_tensor_67);  arg135_1 = add_tensor_67 = None
        copy__default_68: "f32[128]" = torch.ops.aten.copy_.default(arg137_1, add_tensor_68);  arg137_1 = add_tensor_68 = None
        copy__default_69: "f32[512, 128, 1, 1]" = torch.ops.aten.copy_.default(arg139_1, add_tensor_69);  arg139_1 = add_tensor_69 = None
        copy__default_70: "f32[512]" = torch.ops.aten.copy_.default(arg141_1, add_tensor_70);  arg141_1 = add_tensor_70 = None
        copy__default_71: "f32[512]" = torch.ops.aten.copy_.default(arg143_1, add_tensor_71);  arg143_1 = add_tensor_71 = None
        copy__default_72: "f32[256, 512, 1, 1]" = torch.ops.aten.copy_.default(arg145_1, add_tensor_72);  arg145_1 = add_tensor_72 = None
        copy__default_73: "f32[256]" = torch.ops.aten.copy_.default(arg147_1, add_tensor_73);  arg147_1 = add_tensor_73 = None
        copy__default_74: "f32[256]" = torch.ops.aten.copy_.default(arg149_1, add_tensor_74);  arg149_1 = add_tensor_74 = None
        copy__default_75: "f32[256, 256, 3, 3]" = torch.ops.aten.copy_.default(arg151_1, add_tensor_75);  arg151_1 = add_tensor_75 = None
        copy__default_76: "f32[256]" = torch.ops.aten.copy_.default(arg153_1, add_tensor_76);  arg153_1 = add_tensor_76 = None
        copy__default_77: "f32[256]" = torch.ops.aten.copy_.default(arg155_1, add_tensor_77);  arg155_1 = add_tensor_77 = None
        copy__default_78: "f32[1024, 256, 1, 1]" = torch.ops.aten.copy_.default(arg157_1, add_tensor_78);  arg157_1 = add_tensor_78 = None
        copy__default_79: "f32[1024]" = torch.ops.aten.copy_.default(arg159_1, add_tensor_79);  arg159_1 = add_tensor_79 = None
        copy__default_80: "f32[1024]" = torch.ops.aten.copy_.default(arg161_1, add_tensor_80);  arg161_1 = add_tensor_80 = None
        copy__default_81: "f32[1024, 512, 1, 1]" = torch.ops.aten.copy_.default(arg163_1, add_tensor_81);  arg163_1 = add_tensor_81 = None
        copy__default_82: "f32[1024]" = torch.ops.aten.copy_.default(arg165_1, add_tensor_82);  arg165_1 = add_tensor_82 = None
        copy__default_83: "f32[1024]" = torch.ops.aten.copy_.default(arg167_1, add_tensor_83);  arg167_1 = add_tensor_83 = None
        copy__default_84: "f32[256, 1024, 1, 1]" = torch.ops.aten.copy_.default(arg169_1, add_tensor_84);  arg169_1 = add_tensor_84 = None
        copy__default_85: "f32[256]" = torch.ops.aten.copy_.default(arg171_1, add_tensor_85);  arg171_1 = add_tensor_85 = None
        copy__default_86: "f32[256]" = torch.ops.aten.copy_.default(arg173_1, add_tensor_86);  arg173_1 = add_tensor_86 = None
        copy__default_87: "f32[256, 256, 3, 3]" = torch.ops.aten.copy_.default(arg175_1, add_tensor_87);  arg175_1 = add_tensor_87 = None
        copy__default_88: "f32[256]" = torch.ops.aten.copy_.default(arg177_1, add_tensor_88);  arg177_1 = add_tensor_88 = None
        copy__default_89: "f32[256]" = torch.ops.aten.copy_.default(arg179_1, add_tensor_89);  arg179_1 = add_tensor_89 = None
        copy__default_90: "f32[1024, 256, 1, 1]" = torch.ops.aten.copy_.default(arg181_1, add_tensor_90);  arg181_1 = add_tensor_90 = None
        copy__default_91: "f32[1024]" = torch.ops.aten.copy_.default(arg183_1, add_tensor_91);  arg183_1 = add_tensor_91 = None
        copy__default_92: "f32[1024]" = torch.ops.aten.copy_.default(arg185_1, add_tensor_92);  arg185_1 = add_tensor_92 = None
        copy__default_93: "f32[256, 1024, 1, 1]" = torch.ops.aten.copy_.default(arg187_1, add_tensor_93);  arg187_1 = add_tensor_93 = None
        copy__default_94: "f32[256]" = torch.ops.aten.copy_.default(arg189_1, add_tensor_94);  arg189_1 = add_tensor_94 = None
        copy__default_95: "f32[256]" = torch.ops.aten.copy_.default(arg191_1, add_tensor_95);  arg191_1 = add_tensor_95 = None
        copy__default_96: "f32[256, 256, 3, 3]" = torch.ops.aten.copy_.default(arg193_1, add_tensor_96);  arg193_1 = add_tensor_96 = None
        copy__default_97: "f32[256]" = torch.ops.aten.copy_.default(arg195_1, add_tensor_97);  arg195_1 = add_tensor_97 = None
        copy__default_98: "f32[256]" = torch.ops.aten.copy_.default(arg197_1, add_tensor_98);  arg197_1 = add_tensor_98 = None
        copy__default_99: "f32[1024, 256, 1, 1]" = torch.ops.aten.copy_.default(arg199_1, add_tensor_99);  arg199_1 = add_tensor_99 = None
        copy__default_100: "f32[1024]" = torch.ops.aten.copy_.default(arg201_1, add_tensor_100);  arg201_1 = add_tensor_100 = None
        copy__default_101: "f32[1024]" = torch.ops.aten.copy_.default(arg203_1, add_tensor_101);  arg203_1 = add_tensor_101 = None
        copy__default_102: "f32[256, 1024, 1, 1]" = torch.ops.aten.copy_.default(arg205_1, add_tensor_102);  arg205_1 = add_tensor_102 = None
        copy__default_103: "f32[256]" = torch.ops.aten.copy_.default(arg207_1, add_tensor_103);  arg207_1 = add_tensor_103 = None
        copy__default_104: "f32[256]" = torch.ops.aten.copy_.default(arg209_1, add_tensor_104);  arg209_1 = add_tensor_104 = None
        copy__default_105: "f32[256, 256, 3, 3]" = torch.ops.aten.copy_.default(arg211_1, add_tensor_105);  arg211_1 = add_tensor_105 = None
        copy__default_106: "f32[256]" = torch.ops.aten.copy_.default(arg213_1, add_tensor_106);  arg213_1 = add_tensor_106 = None
        copy__default_107: "f32[256]" = torch.ops.aten.copy_.default(arg215_1, add_tensor_107);  arg215_1 = add_tensor_107 = None
        copy__default_108: "f32[1024, 256, 1, 1]" = torch.ops.aten.copy_.default(arg217_1, add_tensor_108);  arg217_1 = add_tensor_108 = None
        copy__default_109: "f32[1024]" = torch.ops.aten.copy_.default(arg219_1, add_tensor_109);  arg219_1 = add_tensor_109 = None
        copy__default_110: "f32[1024]" = torch.ops.aten.copy_.default(arg221_1, add_tensor_110);  arg221_1 = add_tensor_110 = None
        copy__default_111: "f32[256, 1024, 1, 1]" = torch.ops.aten.copy_.default(arg223_1, add_tensor_111);  arg223_1 = add_tensor_111 = None
        copy__default_112: "f32[256]" = torch.ops.aten.copy_.default(arg225_1, add_tensor_112);  arg225_1 = add_tensor_112 = None
        copy__default_113: "f32[256]" = torch.ops.aten.copy_.default(arg227_1, add_tensor_113);  arg227_1 = add_tensor_113 = None
        copy__default_114: "f32[256, 256, 3, 3]" = torch.ops.aten.copy_.default(arg229_1, add_tensor_114);  arg229_1 = add_tensor_114 = None
        copy__default_115: "f32[256]" = torch.ops.aten.copy_.default(arg231_1, add_tensor_115);  arg231_1 = add_tensor_115 = None
        copy__default_116: "f32[256]" = torch.ops.aten.copy_.default(arg233_1, add_tensor_116);  arg233_1 = add_tensor_116 = None
        copy__default_117: "f32[1024, 256, 1, 1]" = torch.ops.aten.copy_.default(arg235_1, add_tensor_117);  arg235_1 = add_tensor_117 = None
        copy__default_118: "f32[1024]" = torch.ops.aten.copy_.default(arg237_1, add_tensor_118);  arg237_1 = add_tensor_118 = None
        copy__default_119: "f32[1024]" = torch.ops.aten.copy_.default(arg239_1, add_tensor_119);  arg239_1 = add_tensor_119 = None
        copy__default_120: "f32[256, 1024, 1, 1]" = torch.ops.aten.copy_.default(arg241_1, add_tensor_120);  arg241_1 = add_tensor_120 = None
        copy__default_121: "f32[256]" = torch.ops.aten.copy_.default(arg243_1, add_tensor_121);  arg243_1 = add_tensor_121 = None
        copy__default_122: "f32[256]" = torch.ops.aten.copy_.default(arg245_1, add_tensor_122);  arg245_1 = add_tensor_122 = None
        copy__default_123: "f32[256, 256, 3, 3]" = torch.ops.aten.copy_.default(arg247_1, add_tensor_123);  arg247_1 = add_tensor_123 = None
        copy__default_124: "f32[256]" = torch.ops.aten.copy_.default(arg249_1, add_tensor_124);  arg249_1 = add_tensor_124 = None
        copy__default_125: "f32[256]" = torch.ops.aten.copy_.default(arg251_1, add_tensor_125);  arg251_1 = add_tensor_125 = None
        copy__default_126: "f32[1024, 256, 1, 1]" = torch.ops.aten.copy_.default(arg253_1, add_tensor_126);  arg253_1 = add_tensor_126 = None
        copy__default_127: "f32[1024]" = torch.ops.aten.copy_.default(arg255_1, add_tensor_127);  arg255_1 = add_tensor_127 = None
        copy__default_128: "f32[1024]" = torch.ops.aten.copy_.default(arg257_1, add_tensor_128);  arg257_1 = add_tensor_128 = None
        copy__default_129: "f32[512, 1024, 1, 1]" = torch.ops.aten.copy_.default(arg259_1, add_tensor_129);  arg259_1 = add_tensor_129 = None
        copy__default_130: "f32[512]" = torch.ops.aten.copy_.default(arg261_1, add_tensor_130);  arg261_1 = add_tensor_130 = None
        copy__default_131: "f32[512]" = torch.ops.aten.copy_.default(arg263_1, add_tensor_131);  arg263_1 = add_tensor_131 = None
        copy__default_132: "f32[512, 512, 3, 3]" = torch.ops.aten.copy_.default(arg265_1, add_tensor_132);  arg265_1 = add_tensor_132 = None
        copy__default_133: "f32[512]" = torch.ops.aten.copy_.default(arg267_1, add_tensor_133);  arg267_1 = add_tensor_133 = None
        copy__default_134: "f32[512]" = torch.ops.aten.copy_.default(arg269_1, add_tensor_134);  arg269_1 = add_tensor_134 = None
        copy__default_135: "f32[2048, 512, 1, 1]" = torch.ops.aten.copy_.default(arg271_1, add_tensor_135);  arg271_1 = add_tensor_135 = None
        copy__default_136: "f32[2048]" = torch.ops.aten.copy_.default(arg273_1, add_tensor_136);  arg273_1 = add_tensor_136 = None
        copy__default_137: "f32[2048]" = torch.ops.aten.copy_.default(arg275_1, add_tensor_137);  arg275_1 = add_tensor_137 = None
        copy__default_138: "f32[2048, 1024, 1, 1]" = torch.ops.aten.copy_.default(arg277_1, add_tensor_138);  arg277_1 = add_tensor_138 = None
        copy__default_139: "f32[2048]" = torch.ops.aten.copy_.default(arg279_1, add_tensor_139);  arg279_1 = add_tensor_139 = None
        copy__default_140: "f32[2048]" = torch.ops.aten.copy_.default(arg281_1, add_tensor_140);  arg281_1 = add_tensor_140 = None
        copy__default_141: "f32[512, 2048, 1, 1]" = torch.ops.aten.copy_.default(arg283_1, add_tensor_141);  arg283_1 = add_tensor_141 = None
        copy__default_142: "f32[512]" = torch.ops.aten.copy_.default(arg285_1, add_tensor_142);  arg285_1 = add_tensor_142 = None
        copy__default_143: "f32[512]" = torch.ops.aten.copy_.default(arg287_1, add_tensor_143);  arg287_1 = add_tensor_143 = None
        copy__default_144: "f32[512, 512, 3, 3]" = torch.ops.aten.copy_.default(arg289_1, add_tensor_144);  arg289_1 = add_tensor_144 = None
        copy__default_145: "f32[512]" = torch.ops.aten.copy_.default(arg291_1, add_tensor_145);  arg291_1 = add_tensor_145 = None
        copy__default_146: "f32[512]" = torch.ops.aten.copy_.default(arg293_1, add_tensor_146);  arg293_1 = add_tensor_146 = None
        copy__default_147: "f32[2048, 512, 1, 1]" = torch.ops.aten.copy_.default(arg295_1, add_tensor_147);  arg295_1 = add_tensor_147 = None
        copy__default_148: "f32[2048]" = torch.ops.aten.copy_.default(arg297_1, add_tensor_148);  arg297_1 = add_tensor_148 = None
        copy__default_149: "f32[2048]" = torch.ops.aten.copy_.default(arg299_1, add_tensor_149);  arg299_1 = add_tensor_149 = None
        copy__default_150: "f32[512, 2048, 1, 1]" = torch.ops.aten.copy_.default(arg301_1, add_tensor_150);  arg301_1 = add_tensor_150 = None
        copy__default_151: "f32[512]" = torch.ops.aten.copy_.default(arg303_1, add_tensor_151);  arg303_1 = add_tensor_151 = None
        copy__default_152: "f32[512]" = torch.ops.aten.copy_.default(arg305_1, add_tensor_152);  arg305_1 = add_tensor_152 = None
        copy__default_153: "f32[512, 512, 3, 3]" = torch.ops.aten.copy_.default(arg307_1, add_tensor_153);  arg307_1 = add_tensor_153 = None
        copy__default_154: "f32[512]" = torch.ops.aten.copy_.default(arg309_1, add_tensor_154);  arg309_1 = add_tensor_154 = None
        copy__default_155: "f32[512]" = torch.ops.aten.copy_.default(arg311_1, add_tensor_155);  arg311_1 = add_tensor_155 = None
        copy__default_156: "f32[2048, 512, 1, 1]" = torch.ops.aten.copy_.default(arg313_1, add_tensor_156);  arg313_1 = add_tensor_156 = None
        copy__default_157: "f32[2048]" = torch.ops.aten.copy_.default(arg315_1, add_tensor_157);  arg315_1 = add_tensor_157 = None
        copy__default_158: "f32[2048]" = torch.ops.aten.copy_.default(arg317_1, add_tensor_158);  arg317_1 = add_tensor_158 = None
        copy__default_159: "f32[128, 2048]" = torch.ops.aten.copy_.default(arg319_1, add_tensor_159);  arg319_1 = add_tensor_159 = None
        copy__default_160: "f32[128]" = torch.ops.aten.copy_.default(arg321_1, add_tensor_160);  arg321_1 = add_tensor_160 = None
        return (div_tensor, copy__default, copy__default_1, copy__default_2, copy__default_3, copy__default_4, copy__default_5, copy__default_6, copy__default_7, copy__default_8, copy__default_9, copy__default_10, copy__default_11, copy__default_12, copy__default_13, copy__default_14, copy__default_15, copy__default_16, copy__default_17, copy__default_18, copy__default_19, copy__default_20, copy__default_21, copy__default_22, copy__default_23, copy__default_24, copy__default_25, copy__default_26, copy__default_27, copy__default_28, copy__default_29, copy__default_30, copy__default_31, copy__default_32, copy__default_33, copy__default_34, copy__default_35, copy__default_36, copy__default_37, copy__default_38, copy__default_39, copy__default_40, copy__default_41, copy__default_42, copy__default_43, copy__default_44, copy__default_45, copy__default_46, copy__default_47, copy__default_48, copy__default_49, copy__default_50, copy__default_51, copy__default_52, copy__default_53, copy__default_54, copy__default_55, copy__default_56, copy__default_57, copy__default_58, copy__default_59, copy__default_60, copy__default_61, copy__default_62, copy__default_63, copy__default_64, copy__default_65, copy__default_66, copy__default_67, copy__default_68, copy__default_69, copy__default_70, copy__default_71, copy__default_72, copy__default_73, copy__default_74, copy__default_75, copy__default_76, copy__default_77, copy__default_78, copy__default_79, copy__default_80, copy__default_81, copy__default_82, copy__default_83, copy__default_84, copy__default_85, copy__default_86, copy__default_87, copy__default_88, copy__default_89, copy__default_90, copy__default_91, copy__default_92, copy__default_93, copy__default_94, copy__default_95, copy__default_96, copy__default_97, copy__default_98, copy__default_99, copy__default_100, copy__default_101, copy__default_102, copy__default_103, copy__default_104, copy__default_105, copy__default_106, copy__default_107, copy__default_108, copy__default_109, copy__default_110, copy__default_111, copy__default_112, copy__default_113, copy__default_114, copy__default_115, copy__default_116, copy__default_117, copy__default_118, copy__default_119, copy__default_120, copy__default_121, copy__default_122, copy__default_123, copy__default_124, copy__default_125, copy__default_126, copy__default_127, copy__default_128, copy__default_129, copy__default_130, copy__default_131, copy__default_132, copy__default_133, copy__default_134, copy__default_135, copy__default_136, copy__default_137, copy__default_138, copy__default_139, copy__default_140, copy__default_141, copy__default_142, copy__default_143, copy__default_144, copy__default_145, copy__default_146, copy__default_147, copy__default_148, copy__default_149, copy__default_150, copy__default_151, copy__default_152, copy__default_153, copy__default_154, copy__default_155, copy__default_156, copy__default_157, copy__default_158, copy__default_159, copy__default_160)



def _default_make_inputs():
    from repro_harness import parse_shapes_config
    return parse_shapes_config(_shapes_config)


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
