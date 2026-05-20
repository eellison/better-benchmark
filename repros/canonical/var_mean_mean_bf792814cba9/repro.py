"""
Standalone repro captured via capture_hook.
Label: inductor_torchbench_perf-4-6-linux.aws.a100_graph24
Pattern hash: bf792814cba9
Shape hash: e698145c
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([], i64, max=2), T([1, 32, 1, 1], f32), T([1, 32, 1, 1], f32), T([32], f32), T([1, 32, 1, 1], f32), T([32], f32), T([], i64, max=2), T([1, 32, 1, 1], f32), T([1, 32, 1, 1], f32), T([32], f32), T([1, 32, 1, 1], f32), T([32], f32), T([], i64, max=2), T([1, 16, 1, 1], f32), T([1, 16, 1, 1], f32), T([16], f32), T([1, 16, 1, 1], f32), T([16], f32), T([], i64, max=2), T([1, 48, 1, 1], f32), T([1, 48, 1, 1], f32), T([48], f32), T([1, 48, 1, 1], f32), T([48], f32), T([], i64, max=2), T([1, 48, 1, 1], f32), T([1, 48, 1, 1], f32), T([48], f32), T([1, 48, 1, 1], f32), T([48], f32), T([], i64, max=2), T([1, 24, 1, 1], f32), T([1, 24, 1, 1], f32), T([24], f32), T([1, 24, 1, 1], f32), T([24], f32), T([], i64, max=2), T([1, 72, 1, 1], f32), T([1, 72, 1, 1], f32), T([72], f32), T([1, 72, 1, 1], f32), T([72], f32), T([], i64, max=2), T([1, 72, 1, 1], f32), T([1, 72, 1, 1], f32), T([72], f32), T([1, 72, 1, 1], f32), T([72], f32), T([], i64, max=2), T([1, 24, 1, 1], f32), T([1, 24, 1, 1], f32), T([24], f32), T([1, 24, 1, 1], f32), T([24], f32), T([], i64, max=2), T([1, 72, 1, 1], f32), T([1, 72, 1, 1], f32), T([72], f32), T([1, 72, 1, 1], f32), T([72], f32), T([], i64, max=2), T([1, 72, 1, 1], f32), T([1, 72, 1, 1], f32), T([72], f32), T([1, 72, 1, 1], f32), T([72], f32), T([], i64, max=2), T([1, 24, 1, 1], f32), T([1, 24, 1, 1], f32), T([24], f32), T([1, 24, 1, 1], f32), T([24], f32), T([], i64, max=2), T([1, 72, 1, 1], f32), T([1, 72, 1, 1], f32), T([72], f32), T([1, 72, 1, 1], f32), T([72], f32), T([], i64, max=2), T([1, 72, 1, 1], f32), T([1, 72, 1, 1], f32), T([72], f32), T([1, 72, 1, 1], f32), T([72], f32), T([], i64, max=2), T([1, 40, 1, 1], f32), T([1, 40, 1, 1], f32), T([40], f32), T([1, 40, 1, 1], f32), T([40], f32), T([], i64, max=2), T([1, 120, 1, 1], f32), T([1, 120, 1, 1], f32), T([120], f32), T([1, 120, 1, 1], f32), T([120], f32), T([], i64, max=2), T([1, 120, 1, 1], f32), T([1, 120, 1, 1], f32), T([120], f32), T([1, 120, 1, 1], f32), T([120], f32), T([], i64, max=2), T([1, 40, 1, 1], f32), T([1, 40, 1, 1], f32), T([40], f32), T([1, 40, 1, 1], f32), T([40], f32), T([], i64, max=2), T([1, 120, 1, 1], f32), T([1, 120, 1, 1], f32), T([120], f32), T([1, 120, 1, 1], f32), T([120], f32), T([], i64, max=2), T([1, 120, 1, 1], f32), T([1, 120, 1, 1], f32), T([120], f32), T([1, 120, 1, 1], f32), T([120], f32), T([], i64, max=2), T([1, 40, 1, 1], f32), T([1, 40, 1, 1], f32), T([40], f32), T([1, 40, 1, 1], f32), T([40], f32), T([], i64, max=2), T([1, 240, 1, 1], f32), T([1, 240, 1, 1], f32), T([240], f32), T([1, 240, 1, 1], f32), T([240], f32), T([], i64, max=2), T([1, 240, 1, 1], f32), T([1, 240, 1, 1], f32), T([240], f32), T([1, 240, 1, 1], f32), T([240], f32), T([], i64, max=2), T([1, 80, 1, 1], f32), T([1, 80, 1, 1], f32), T([80], f32), T([1, 80, 1, 1], f32), T([80], f32), T([], i64, max=2), T([1, 480, 1, 1], f32), T([1, 480, 1, 1], f32), T([480], f32), T([1, 480, 1, 1], f32), T([480], f32), T([], i64, max=2), T([1, 480, 1, 1], f32), T([1, 480, 1, 1], f32), T([480], f32), T([1, 480, 1, 1], f32), T([480], f32), T([], i64, max=2), T([1, 80, 1, 1], f32), T([1, 80, 1, 1], f32), T([80], f32), T([1, 80, 1, 1], f32), T([80], f32), T([], i64, max=2), T([1, 480, 1, 1], f32), T([1, 480, 1, 1], f32), T([480], f32), T([1, 480, 1, 1], f32), T([480], f32), T([], i64, max=2), T([1, 480, 1, 1], f32), T([1, 480, 1, 1], f32), T([480], f32), T([1, 480, 1, 1], f32), T([480], f32), T([], i64, max=2), T([1, 80, 1, 1], f32), T([1, 80, 1, 1], f32), T([80], f32), T([1, 80, 1, 1], f32), T([80], f32), T([], i64, max=2), T([1, 480, 1, 1], f32), T([1, 480, 1, 1], f32), T([480], f32), T([1, 480, 1, 1], f32), T([480], f32), T([], i64, max=2), T([1, 480, 1, 1], f32), T([1, 480, 1, 1], f32), T([480], f32), T([1, 480, 1, 1], f32), T([480], f32), T([], i64, max=2), T([1, 96, 1, 1], f32), T([1, 96, 1, 1], f32), T([96], f32), T([1, 96, 1, 1], f32), T([96], f32), T([], i64, max=2), T([1, 576, 1, 1], f32), T([1, 576, 1, 1], f32), T([576], f32), T([1, 576, 1, 1], f32), T([576], f32), T([], i64, max=2), T([1, 576, 1, 1], f32), T([1, 576, 1, 1], f32), T([576], f32), T([1, 576, 1, 1], f32), T([576], f32), T([], i64, max=2), T([1, 96, 1, 1], f32), T([1, 96, 1, 1], f32), T([96], f32), T([1, 96, 1, 1], f32), T([96], f32), T([], i64, max=2), T([1, 576, 1, 1], f32), T([1, 576, 1, 1], f32), T([576], f32), T([1, 576, 1, 1], f32), T([576], f32), T([], i64, max=2), T([1, 576, 1, 1], f32), T([1, 576, 1, 1], f32), T([576], f32), T([1, 576, 1, 1], f32), T([576], f32), T([], i64, max=2), T([1, 192, 1, 1], f32), T([1, 192, 1, 1], f32), T([192], f32), T([1, 192, 1, 1], f32), T([192], f32), T([], i64, max=2), T([1, 1152, 1, 1], f32), T([1, 1152, 1, 1], f32), T([1152], f32), T([1, 1152, 1, 1], f32), T([1152], f32), T([], i64, max=2), T([1, 1152, 1, 1], f32), T([1, 1152, 1, 1], f32), T([1152], f32), T([1, 1152, 1, 1], f32), T([1152], f32), T([], i64, max=2), T([1, 192, 1, 1], f32), T([1, 192, 1, 1], f32), T([192], f32), T([1, 192, 1, 1], f32), T([192], f32), T([], i64, max=2), T([1, 1152, 1, 1], f32), T([1, 1152, 1, 1], f32), T([1152], f32), T([1, 1152, 1, 1], f32), T([1152], f32), T([], i64, max=2), T([1, 1152, 1, 1], f32), T([1, 1152, 1, 1], f32), T([1152], f32), T([1, 1152, 1, 1], f32), T([1152], f32), T([], i64, max=2), T([1, 192, 1, 1], f32), T([1, 192, 1, 1], f32), T([192], f32), T([1, 192, 1, 1], f32), T([192], f32), T([], i64, max=2), T([1, 1152, 1, 1], f32), T([1, 1152, 1, 1], f32), T([1152], f32), T([1, 1152, 1, 1], f32), T([1152], f32), T([], i64, max=2), T([1, 1152, 1, 1], f32), T([1, 1152, 1, 1], f32), T([1152], f32), T([1, 1152, 1, 1], f32), T([1152], f32), T([], i64, max=2), T([1, 192, 1, 1], f32), T([1, 192, 1, 1], f32), T([192], f32), T([1, 192, 1, 1], f32), T([192], f32), T([], i64, max=2), T([1, 1152, 1, 1], f32), T([1, 1152, 1, 1], f32), T([1152], f32), T([1, 1152, 1, 1], f32), T([1152], f32), T([], i64, max=2), T([1, 1152, 1, 1], f32), T([1, 1152, 1, 1], f32), T([1152], f32), T([1, 1152, 1, 1], f32), T([1152], f32), T([], i64, max=2), T([1, 320, 1, 1], f32), T([1, 320, 1, 1], f32), T([320], f32), T([1, 320, 1, 1], f32), T([320], f32), T([], i64, max=2), T([32, 1280, 7, 7], f16), T([1280], f32), T([1280], f32), T([1280], f32), T([1280], f32), T([1000], f32), T([1000, 1280], f32))"

class Repro(torch.nn.Module):
    def forward(self, arg2_1: "i64[]", getitem_1: "f32[1, 32, 1, 1]", rsqrt: "f32[1, 32, 1, 1]", arg3_1: "f32[32]", getitem: "f32[1, 32, 1, 1]", arg4_1: "f32[32]", arg8_1: "i64[]", getitem_3: "f32[1, 32, 1, 1]", rsqrt_1: "f32[1, 32, 1, 1]", arg9_1: "f32[32]", getitem_2: "f32[1, 32, 1, 1]", arg10_1: "f32[32]", arg14_1: "i64[]", getitem_5: "f32[1, 16, 1, 1]", rsqrt_2: "f32[1, 16, 1, 1]", arg15_1: "f32[16]", getitem_4: "f32[1, 16, 1, 1]", arg16_1: "f32[16]", arg20_1: "i64[]", getitem_7: "f32[1, 48, 1, 1]", rsqrt_3: "f32[1, 48, 1, 1]", arg21_1: "f32[48]", getitem_6: "f32[1, 48, 1, 1]", arg22_1: "f32[48]", arg26_1: "i64[]", getitem_9: "f32[1, 48, 1, 1]", rsqrt_4: "f32[1, 48, 1, 1]", arg27_1: "f32[48]", getitem_8: "f32[1, 48, 1, 1]", arg28_1: "f32[48]", arg32_1: "i64[]", getitem_11: "f32[1, 24, 1, 1]", rsqrt_5: "f32[1, 24, 1, 1]", arg33_1: "f32[24]", getitem_10: "f32[1, 24, 1, 1]", arg34_1: "f32[24]", arg38_1: "i64[]", getitem_13: "f32[1, 72, 1, 1]", rsqrt_6: "f32[1, 72, 1, 1]", arg39_1: "f32[72]", getitem_12: "f32[1, 72, 1, 1]", arg40_1: "f32[72]", arg44_1: "i64[]", getitem_15: "f32[1, 72, 1, 1]", rsqrt_7: "f32[1, 72, 1, 1]", arg45_1: "f32[72]", getitem_14: "f32[1, 72, 1, 1]", arg46_1: "f32[72]", arg50_1: "i64[]", getitem_17: "f32[1, 24, 1, 1]", rsqrt_8: "f32[1, 24, 1, 1]", arg51_1: "f32[24]", getitem_16: "f32[1, 24, 1, 1]", arg52_1: "f32[24]", arg56_1: "i64[]", getitem_19: "f32[1, 72, 1, 1]", rsqrt_9: "f32[1, 72, 1, 1]", arg57_1: "f32[72]", getitem_18: "f32[1, 72, 1, 1]", arg58_1: "f32[72]", arg62_1: "i64[]", getitem_21: "f32[1, 72, 1, 1]", rsqrt_10: "f32[1, 72, 1, 1]", arg63_1: "f32[72]", getitem_20: "f32[1, 72, 1, 1]", arg64_1: "f32[72]", arg68_1: "i64[]", getitem_23: "f32[1, 24, 1, 1]", rsqrt_11: "f32[1, 24, 1, 1]", arg69_1: "f32[24]", getitem_22: "f32[1, 24, 1, 1]", arg70_1: "f32[24]", arg74_1: "i64[]", getitem_25: "f32[1, 72, 1, 1]", rsqrt_12: "f32[1, 72, 1, 1]", arg75_1: "f32[72]", getitem_24: "f32[1, 72, 1, 1]", arg76_1: "f32[72]", arg80_1: "i64[]", getitem_27: "f32[1, 72, 1, 1]", rsqrt_13: "f32[1, 72, 1, 1]", arg81_1: "f32[72]", getitem_26: "f32[1, 72, 1, 1]", arg82_1: "f32[72]", arg86_1: "i64[]", getitem_29: "f32[1, 40, 1, 1]", rsqrt_14: "f32[1, 40, 1, 1]", arg87_1: "f32[40]", getitem_28: "f32[1, 40, 1, 1]", arg88_1: "f32[40]", arg92_1: "i64[]", getitem_31: "f32[1, 120, 1, 1]", rsqrt_15: "f32[1, 120, 1, 1]", arg93_1: "f32[120]", getitem_30: "f32[1, 120, 1, 1]", arg94_1: "f32[120]", arg98_1: "i64[]", getitem_33: "f32[1, 120, 1, 1]", rsqrt_16: "f32[1, 120, 1, 1]", arg99_1: "f32[120]", getitem_32: "f32[1, 120, 1, 1]", arg100_1: "f32[120]", arg104_1: "i64[]", getitem_35: "f32[1, 40, 1, 1]", rsqrt_17: "f32[1, 40, 1, 1]", arg105_1: "f32[40]", getitem_34: "f32[1, 40, 1, 1]", arg106_1: "f32[40]", arg110_1: "i64[]", getitem_37: "f32[1, 120, 1, 1]", rsqrt_18: "f32[1, 120, 1, 1]", arg111_1: "f32[120]", getitem_36: "f32[1, 120, 1, 1]", arg112_1: "f32[120]", arg116_1: "i64[]", getitem_39: "f32[1, 120, 1, 1]", rsqrt_19: "f32[1, 120, 1, 1]", arg117_1: "f32[120]", getitem_38: "f32[1, 120, 1, 1]", arg118_1: "f32[120]", arg122_1: "i64[]", getitem_41: "f32[1, 40, 1, 1]", rsqrt_20: "f32[1, 40, 1, 1]", arg123_1: "f32[40]", getitem_40: "f32[1, 40, 1, 1]", arg124_1: "f32[40]", arg128_1: "i64[]", getitem_43: "f32[1, 240, 1, 1]", rsqrt_21: "f32[1, 240, 1, 1]", arg129_1: "f32[240]", getitem_42: "f32[1, 240, 1, 1]", arg130_1: "f32[240]", arg134_1: "i64[]", getitem_45: "f32[1, 240, 1, 1]", rsqrt_22: "f32[1, 240, 1, 1]", arg135_1: "f32[240]", getitem_44: "f32[1, 240, 1, 1]", arg136_1: "f32[240]", arg140_1: "i64[]", getitem_47: "f32[1, 80, 1, 1]", rsqrt_23: "f32[1, 80, 1, 1]", arg141_1: "f32[80]", getitem_46: "f32[1, 80, 1, 1]", arg142_1: "f32[80]", arg146_1: "i64[]", getitem_49: "f32[1, 480, 1, 1]", rsqrt_24: "f32[1, 480, 1, 1]", arg147_1: "f32[480]", getitem_48: "f32[1, 480, 1, 1]", arg148_1: "f32[480]", arg152_1: "i64[]", getitem_51: "f32[1, 480, 1, 1]", rsqrt_25: "f32[1, 480, 1, 1]", arg153_1: "f32[480]", getitem_50: "f32[1, 480, 1, 1]", arg154_1: "f32[480]", arg158_1: "i64[]", getitem_53: "f32[1, 80, 1, 1]", rsqrt_26: "f32[1, 80, 1, 1]", arg159_1: "f32[80]", getitem_52: "f32[1, 80, 1, 1]", arg160_1: "f32[80]", arg164_1: "i64[]", getitem_55: "f32[1, 480, 1, 1]", rsqrt_27: "f32[1, 480, 1, 1]", arg165_1: "f32[480]", getitem_54: "f32[1, 480, 1, 1]", arg166_1: "f32[480]", arg170_1: "i64[]", getitem_57: "f32[1, 480, 1, 1]", rsqrt_28: "f32[1, 480, 1, 1]", arg171_1: "f32[480]", getitem_56: "f32[1, 480, 1, 1]", arg172_1: "f32[480]", arg176_1: "i64[]", getitem_59: "f32[1, 80, 1, 1]", rsqrt_29: "f32[1, 80, 1, 1]", arg177_1: "f32[80]", getitem_58: "f32[1, 80, 1, 1]", arg178_1: "f32[80]", arg182_1: "i64[]", getitem_61: "f32[1, 480, 1, 1]", rsqrt_30: "f32[1, 480, 1, 1]", arg183_1: "f32[480]", getitem_60: "f32[1, 480, 1, 1]", arg184_1: "f32[480]", arg188_1: "i64[]", getitem_63: "f32[1, 480, 1, 1]", rsqrt_31: "f32[1, 480, 1, 1]", arg189_1: "f32[480]", getitem_62: "f32[1, 480, 1, 1]", arg190_1: "f32[480]", arg194_1: "i64[]", getitem_65: "f32[1, 96, 1, 1]", rsqrt_32: "f32[1, 96, 1, 1]", arg195_1: "f32[96]", getitem_64: "f32[1, 96, 1, 1]", arg196_1: "f32[96]", arg200_1: "i64[]", getitem_67: "f32[1, 576, 1, 1]", rsqrt_33: "f32[1, 576, 1, 1]", arg201_1: "f32[576]", getitem_66: "f32[1, 576, 1, 1]", arg202_1: "f32[576]", arg206_1: "i64[]", getitem_69: "f32[1, 576, 1, 1]", rsqrt_34: "f32[1, 576, 1, 1]", arg207_1: "f32[576]", getitem_68: "f32[1, 576, 1, 1]", arg208_1: "f32[576]", arg212_1: "i64[]", getitem_71: "f32[1, 96, 1, 1]", rsqrt_35: "f32[1, 96, 1, 1]", arg213_1: "f32[96]", getitem_70: "f32[1, 96, 1, 1]", arg214_1: "f32[96]", arg218_1: "i64[]", getitem_73: "f32[1, 576, 1, 1]", rsqrt_36: "f32[1, 576, 1, 1]", arg219_1: "f32[576]", getitem_72: "f32[1, 576, 1, 1]", arg220_1: "f32[576]", arg224_1: "i64[]", getitem_75: "f32[1, 576, 1, 1]", rsqrt_37: "f32[1, 576, 1, 1]", arg225_1: "f32[576]", getitem_74: "f32[1, 576, 1, 1]", arg226_1: "f32[576]", arg230_1: "i64[]", getitem_77: "f32[1, 192, 1, 1]", rsqrt_38: "f32[1, 192, 1, 1]", arg231_1: "f32[192]", getitem_76: "f32[1, 192, 1, 1]", arg232_1: "f32[192]", arg236_1: "i64[]", getitem_79: "f32[1, 1152, 1, 1]", rsqrt_39: "f32[1, 1152, 1, 1]", arg237_1: "f32[1152]", getitem_78: "f32[1, 1152, 1, 1]", arg238_1: "f32[1152]", arg242_1: "i64[]", getitem_81: "f32[1, 1152, 1, 1]", rsqrt_40: "f32[1, 1152, 1, 1]", arg243_1: "f32[1152]", getitem_80: "f32[1, 1152, 1, 1]", arg244_1: "f32[1152]", arg248_1: "i64[]", getitem_83: "f32[1, 192, 1, 1]", rsqrt_41: "f32[1, 192, 1, 1]", arg249_1: "f32[192]", getitem_82: "f32[1, 192, 1, 1]", arg250_1: "f32[192]", arg254_1: "i64[]", getitem_85: "f32[1, 1152, 1, 1]", rsqrt_42: "f32[1, 1152, 1, 1]", arg255_1: "f32[1152]", getitem_84: "f32[1, 1152, 1, 1]", arg256_1: "f32[1152]", arg260_1: "i64[]", getitem_87: "f32[1, 1152, 1, 1]", rsqrt_43: "f32[1, 1152, 1, 1]", arg261_1: "f32[1152]", getitem_86: "f32[1, 1152, 1, 1]", arg262_1: "f32[1152]", arg266_1: "i64[]", getitem_89: "f32[1, 192, 1, 1]", rsqrt_44: "f32[1, 192, 1, 1]", arg267_1: "f32[192]", getitem_88: "f32[1, 192, 1, 1]", arg268_1: "f32[192]", arg272_1: "i64[]", getitem_91: "f32[1, 1152, 1, 1]", rsqrt_45: "f32[1, 1152, 1, 1]", arg273_1: "f32[1152]", getitem_90: "f32[1, 1152, 1, 1]", arg274_1: "f32[1152]", arg278_1: "i64[]", getitem_93: "f32[1, 1152, 1, 1]", rsqrt_46: "f32[1, 1152, 1, 1]", arg279_1: "f32[1152]", getitem_92: "f32[1, 1152, 1, 1]", arg280_1: "f32[1152]", arg284_1: "i64[]", getitem_95: "f32[1, 192, 1, 1]", rsqrt_47: "f32[1, 192, 1, 1]", arg285_1: "f32[192]", getitem_94: "f32[1, 192, 1, 1]", arg286_1: "f32[192]", arg290_1: "i64[]", getitem_97: "f32[1, 1152, 1, 1]", rsqrt_48: "f32[1, 1152, 1, 1]", arg291_1: "f32[1152]", getitem_96: "f32[1, 1152, 1, 1]", arg292_1: "f32[1152]", arg296_1: "i64[]", getitem_99: "f32[1, 1152, 1, 1]", rsqrt_49: "f32[1, 1152, 1, 1]", arg297_1: "f32[1152]", getitem_98: "f32[1, 1152, 1, 1]", arg298_1: "f32[1152]", arg302_1: "i64[]", getitem_101: "f32[1, 320, 1, 1]", rsqrt_50: "f32[1, 320, 1, 1]", arg303_1: "f32[320]", getitem_100: "f32[1, 320, 1, 1]", arg304_1: "f32[320]", arg308_1: "i64[]", convolution_51: "f16[32, 1280, 7, 7]", arg309_1: "f32[1280]", arg310_1: "f32[1280]", arg311_1: "f32[1280]", arg312_1: "f32[1280]", arg314_1: "f32[1000]", arg313_1: "f32[1000, 1280]"):
        # No stacktrace found for following nodes
        add_tensor: "i64[]" = torch.ops.aten.add.Tensor(arg2_1, 1)
        squeeze_dims: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        squeeze_dims_1: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
        mul_tensor: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_dims, 0.00029999999999996696)
        mul_tensor_1: "f32[32]" = torch.ops.aten.mul.Tensor(arg3_1, 0.9997)
        add_tensor_1: "f32[32]" = torch.ops.aten.add.Tensor(mul_tensor, mul_tensor_1);  mul_tensor = mul_tensor_1 = None
        squeeze_dims_2: "f32[32]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_tensor_2: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_dims_2, 1.0000024912370735);  squeeze_dims_2 = None
        mul_tensor_3: "f32[32]" = torch.ops.aten.mul.Tensor(mul_tensor_2, 0.00029999999999996696);  mul_tensor_2 = None
        mul_tensor_4: "f32[32]" = torch.ops.aten.mul.Tensor(arg4_1, 0.9997)
        add_tensor_2: "f32[32]" = torch.ops.aten.add.Tensor(mul_tensor_3, mul_tensor_4);  mul_tensor_3 = mul_tensor_4 = None
        add_tensor_3: "i64[]" = torch.ops.aten.add.Tensor(arg8_1, 1)
        squeeze_dims_3: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
        squeeze_dims_4: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
        mul_tensor_5: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_dims_3, 0.00029999999999996696)
        mul_tensor_6: "f32[32]" = torch.ops.aten.mul.Tensor(arg9_1, 0.9997)
        add_tensor_4: "f32[32]" = torch.ops.aten.add.Tensor(mul_tensor_5, mul_tensor_6);  mul_tensor_5 = mul_tensor_6 = None
        squeeze_dims_5: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
        mul_tensor_7: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_dims_5, 1.0000024912370735);  squeeze_dims_5 = None
        mul_tensor_8: "f32[32]" = torch.ops.aten.mul.Tensor(mul_tensor_7, 0.00029999999999996696);  mul_tensor_7 = None
        mul_tensor_9: "f32[32]" = torch.ops.aten.mul.Tensor(arg10_1, 0.9997)
        add_tensor_5: "f32[32]" = torch.ops.aten.add.Tensor(mul_tensor_8, mul_tensor_9);  mul_tensor_8 = mul_tensor_9 = None
        add_tensor_6: "i64[]" = torch.ops.aten.add.Tensor(arg14_1, 1)
        squeeze_dims_6: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
        squeeze_dims_7: "f32[16]" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
        mul_tensor_10: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_dims_6, 0.00029999999999996696)
        mul_tensor_11: "f32[16]" = torch.ops.aten.mul.Tensor(arg15_1, 0.9997)
        add_tensor_7: "f32[16]" = torch.ops.aten.add.Tensor(mul_tensor_10, mul_tensor_11);  mul_tensor_10 = mul_tensor_11 = None
        squeeze_dims_8: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
        mul_tensor_12: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_dims_8, 1.0000024912370735);  squeeze_dims_8 = None
        mul_tensor_13: "f32[16]" = torch.ops.aten.mul.Tensor(mul_tensor_12, 0.00029999999999996696);  mul_tensor_12 = None
        mul_tensor_14: "f32[16]" = torch.ops.aten.mul.Tensor(arg16_1, 0.9997)
        add_tensor_8: "f32[16]" = torch.ops.aten.add.Tensor(mul_tensor_13, mul_tensor_14);  mul_tensor_13 = mul_tensor_14 = None
        add_tensor_9: "i64[]" = torch.ops.aten.add.Tensor(arg20_1, 1)
        squeeze_dims_9: "f32[48]" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2, 3]);  getitem_7 = None
        squeeze_dims_10: "f32[48]" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
        mul_tensor_15: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_dims_9, 0.00029999999999996696)
        mul_tensor_16: "f32[48]" = torch.ops.aten.mul.Tensor(arg21_1, 0.9997)
        add_tensor_10: "f32[48]" = torch.ops.aten.add.Tensor(mul_tensor_15, mul_tensor_16);  mul_tensor_15 = mul_tensor_16 = None
        squeeze_dims_11: "f32[48]" = torch.ops.aten.squeeze.dims(getitem_6, [0, 2, 3]);  getitem_6 = None
        mul_tensor_17: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_dims_11, 1.0000024912370735);  squeeze_dims_11 = None
        mul_tensor_18: "f32[48]" = torch.ops.aten.mul.Tensor(mul_tensor_17, 0.00029999999999996696);  mul_tensor_17 = None
        mul_tensor_19: "f32[48]" = torch.ops.aten.mul.Tensor(arg22_1, 0.9997)
        add_tensor_11: "f32[48]" = torch.ops.aten.add.Tensor(mul_tensor_18, mul_tensor_19);  mul_tensor_18 = mul_tensor_19 = None
        add_tensor_12: "i64[]" = torch.ops.aten.add.Tensor(arg26_1, 1)
        squeeze_dims_12: "f32[48]" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3]);  getitem_9 = None
        squeeze_dims_13: "f32[48]" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
        mul_tensor_20: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_dims_12, 0.00029999999999996696)
        mul_tensor_21: "f32[48]" = torch.ops.aten.mul.Tensor(arg27_1, 0.9997)
        add_tensor_13: "f32[48]" = torch.ops.aten.add.Tensor(mul_tensor_20, mul_tensor_21);  mul_tensor_20 = mul_tensor_21 = None
        squeeze_dims_14: "f32[48]" = torch.ops.aten.squeeze.dims(getitem_8, [0, 2, 3]);  getitem_8 = None
        mul_tensor_22: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_dims_14, 1.00000996502277);  squeeze_dims_14 = None
        mul_tensor_23: "f32[48]" = torch.ops.aten.mul.Tensor(mul_tensor_22, 0.00029999999999996696);  mul_tensor_22 = None
        mul_tensor_24: "f32[48]" = torch.ops.aten.mul.Tensor(arg28_1, 0.9997)
        add_tensor_14: "f32[48]" = torch.ops.aten.add.Tensor(mul_tensor_23, mul_tensor_24);  mul_tensor_23 = mul_tensor_24 = None
        add_tensor_15: "i64[]" = torch.ops.aten.add.Tensor(arg32_1, 1)
        squeeze_dims_15: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3]);  getitem_11 = None
        squeeze_dims_16: "f32[24]" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
        mul_tensor_25: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_dims_15, 0.00029999999999996696)
        mul_tensor_26: "f32[24]" = torch.ops.aten.mul.Tensor(arg33_1, 0.9997)
        add_tensor_16: "f32[24]" = torch.ops.aten.add.Tensor(mul_tensor_25, mul_tensor_26);  mul_tensor_25 = mul_tensor_26 = None
        squeeze_dims_17: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_10, [0, 2, 3]);  getitem_10 = None
        mul_tensor_27: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_dims_17, 1.00000996502277);  squeeze_dims_17 = None
        mul_tensor_28: "f32[24]" = torch.ops.aten.mul.Tensor(mul_tensor_27, 0.00029999999999996696);  mul_tensor_27 = None
        mul_tensor_29: "f32[24]" = torch.ops.aten.mul.Tensor(arg34_1, 0.9997)
        add_tensor_17: "f32[24]" = torch.ops.aten.add.Tensor(mul_tensor_28, mul_tensor_29);  mul_tensor_28 = mul_tensor_29 = None
        add_tensor_18: "i64[]" = torch.ops.aten.add.Tensor(arg38_1, 1)
        squeeze_dims_18: "f32[72]" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
        squeeze_dims_19: "f32[72]" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
        mul_tensor_30: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_dims_18, 0.00029999999999996696)
        mul_tensor_31: "f32[72]" = torch.ops.aten.mul.Tensor(arg39_1, 0.9997)
        add_tensor_19: "f32[72]" = torch.ops.aten.add.Tensor(mul_tensor_30, mul_tensor_31);  mul_tensor_30 = mul_tensor_31 = None
        squeeze_dims_20: "f32[72]" = torch.ops.aten.squeeze.dims(getitem_12, [0, 2, 3]);  getitem_12 = None
        mul_tensor_32: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_dims_20, 1.00000996502277);  squeeze_dims_20 = None
        mul_tensor_33: "f32[72]" = torch.ops.aten.mul.Tensor(mul_tensor_32, 0.00029999999999996696);  mul_tensor_32 = None
        mul_tensor_34: "f32[72]" = torch.ops.aten.mul.Tensor(arg40_1, 0.9997)
        add_tensor_20: "f32[72]" = torch.ops.aten.add.Tensor(mul_tensor_33, mul_tensor_34);  mul_tensor_33 = mul_tensor_34 = None
        add_tensor_21: "i64[]" = torch.ops.aten.add.Tensor(arg44_1, 1)
        squeeze_dims_21: "f32[72]" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3]);  getitem_15 = None
        squeeze_dims_22: "f32[72]" = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2, 3]);  rsqrt_7 = None
        mul_tensor_35: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_dims_21, 0.00029999999999996696)
        mul_tensor_36: "f32[72]" = torch.ops.aten.mul.Tensor(arg45_1, 0.9997)
        add_tensor_22: "f32[72]" = torch.ops.aten.add.Tensor(mul_tensor_35, mul_tensor_36);  mul_tensor_35 = mul_tensor_36 = None
        squeeze_dims_23: "f32[72]" = torch.ops.aten.squeeze.dims(getitem_14, [0, 2, 3]);  getitem_14 = None
        mul_tensor_37: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_dims_23, 1.00000996502277);  squeeze_dims_23 = None
        mul_tensor_38: "f32[72]" = torch.ops.aten.mul.Tensor(mul_tensor_37, 0.00029999999999996696);  mul_tensor_37 = None
        mul_tensor_39: "f32[72]" = torch.ops.aten.mul.Tensor(arg46_1, 0.9997)
        add_tensor_23: "f32[72]" = torch.ops.aten.add.Tensor(mul_tensor_38, mul_tensor_39);  mul_tensor_38 = mul_tensor_39 = None
        add_tensor_24: "i64[]" = torch.ops.aten.add.Tensor(arg50_1, 1)
        squeeze_dims_24: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_17, [0, 2, 3]);  getitem_17 = None
        squeeze_dims_25: "f32[24]" = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2, 3]);  rsqrt_8 = None
        mul_tensor_40: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_dims_24, 0.00029999999999996696)
        mul_tensor_41: "f32[24]" = torch.ops.aten.mul.Tensor(arg51_1, 0.9997)
        add_tensor_25: "f32[24]" = torch.ops.aten.add.Tensor(mul_tensor_40, mul_tensor_41);  mul_tensor_40 = mul_tensor_41 = None
        squeeze_dims_26: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_16, [0, 2, 3]);  getitem_16 = None
        mul_tensor_42: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_dims_26, 1.00000996502277);  squeeze_dims_26 = None
        mul_tensor_43: "f32[24]" = torch.ops.aten.mul.Tensor(mul_tensor_42, 0.00029999999999996696);  mul_tensor_42 = None
        mul_tensor_44: "f32[24]" = torch.ops.aten.mul.Tensor(arg52_1, 0.9997)
        add_tensor_26: "f32[24]" = torch.ops.aten.add.Tensor(mul_tensor_43, mul_tensor_44);  mul_tensor_43 = mul_tensor_44 = None
        add_tensor_27: "i64[]" = torch.ops.aten.add.Tensor(arg56_1, 1)
        squeeze_dims_27: "f32[72]" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3]);  getitem_19 = None
        squeeze_dims_28: "f32[72]" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
        mul_tensor_45: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_dims_27, 0.00029999999999996696)
        mul_tensor_46: "f32[72]" = torch.ops.aten.mul.Tensor(arg57_1, 0.9997)
        add_tensor_28: "f32[72]" = torch.ops.aten.add.Tensor(mul_tensor_45, mul_tensor_46);  mul_tensor_45 = mul_tensor_46 = None
        squeeze_dims_29: "f32[72]" = torch.ops.aten.squeeze.dims(getitem_18, [0, 2, 3]);  getitem_18 = None
        mul_tensor_47: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_dims_29, 1.00000996502277);  squeeze_dims_29 = None
        mul_tensor_48: "f32[72]" = torch.ops.aten.mul.Tensor(mul_tensor_47, 0.00029999999999996696);  mul_tensor_47 = None
        mul_tensor_49: "f32[72]" = torch.ops.aten.mul.Tensor(arg58_1, 0.9997)
        add_tensor_29: "f32[72]" = torch.ops.aten.add.Tensor(mul_tensor_48, mul_tensor_49);  mul_tensor_48 = mul_tensor_49 = None
        add_tensor_30: "i64[]" = torch.ops.aten.add.Tensor(arg62_1, 1)
        squeeze_dims_30: "f32[72]" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
        squeeze_dims_31: "f32[72]" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
        mul_tensor_50: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_dims_30, 0.00029999999999996696)
        mul_tensor_51: "f32[72]" = torch.ops.aten.mul.Tensor(arg63_1, 0.9997)
        add_tensor_31: "f32[72]" = torch.ops.aten.add.Tensor(mul_tensor_50, mul_tensor_51);  mul_tensor_50 = mul_tensor_51 = None
        squeeze_dims_32: "f32[72]" = torch.ops.aten.squeeze.dims(getitem_20, [0, 2, 3]);  getitem_20 = None
        mul_tensor_52: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_dims_32, 1.00000996502277);  squeeze_dims_32 = None
        mul_tensor_53: "f32[72]" = torch.ops.aten.mul.Tensor(mul_tensor_52, 0.00029999999999996696);  mul_tensor_52 = None
        mul_tensor_54: "f32[72]" = torch.ops.aten.mul.Tensor(arg64_1, 0.9997)
        add_tensor_32: "f32[72]" = torch.ops.aten.add.Tensor(mul_tensor_53, mul_tensor_54);  mul_tensor_53 = mul_tensor_54 = None
        add_tensor_33: "i64[]" = torch.ops.aten.add.Tensor(arg68_1, 1)
        squeeze_dims_33: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_23, [0, 2, 3]);  getitem_23 = None
        squeeze_dims_34: "f32[24]" = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2, 3]);  rsqrt_11 = None
        mul_tensor_55: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_dims_33, 0.00029999999999996696)
        mul_tensor_56: "f32[24]" = torch.ops.aten.mul.Tensor(arg69_1, 0.9997)
        add_tensor_34: "f32[24]" = torch.ops.aten.add.Tensor(mul_tensor_55, mul_tensor_56);  mul_tensor_55 = mul_tensor_56 = None
        squeeze_dims_35: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_22, [0, 2, 3]);  getitem_22 = None
        mul_tensor_57: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_dims_35, 1.00000996502277);  squeeze_dims_35 = None
        mul_tensor_58: "f32[24]" = torch.ops.aten.mul.Tensor(mul_tensor_57, 0.00029999999999996696);  mul_tensor_57 = None
        mul_tensor_59: "f32[24]" = torch.ops.aten.mul.Tensor(arg70_1, 0.9997)
        add_tensor_35: "f32[24]" = torch.ops.aten.add.Tensor(mul_tensor_58, mul_tensor_59);  mul_tensor_58 = mul_tensor_59 = None
        add_tensor_36: "i64[]" = torch.ops.aten.add.Tensor(arg74_1, 1)
        squeeze_dims_36: "f32[72]" = torch.ops.aten.squeeze.dims(getitem_25, [0, 2, 3]);  getitem_25 = None
        squeeze_dims_37: "f32[72]" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
        mul_tensor_60: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_dims_36, 0.00029999999999996696)
        mul_tensor_61: "f32[72]" = torch.ops.aten.mul.Tensor(arg75_1, 0.9997)
        add_tensor_37: "f32[72]" = torch.ops.aten.add.Tensor(mul_tensor_60, mul_tensor_61);  mul_tensor_60 = mul_tensor_61 = None
        squeeze_dims_38: "f32[72]" = torch.ops.aten.squeeze.dims(getitem_24, [0, 2, 3]);  getitem_24 = None
        mul_tensor_62: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_dims_38, 1.00000996502277);  squeeze_dims_38 = None
        mul_tensor_63: "f32[72]" = torch.ops.aten.mul.Tensor(mul_tensor_62, 0.00029999999999996696);  mul_tensor_62 = None
        mul_tensor_64: "f32[72]" = torch.ops.aten.mul.Tensor(arg76_1, 0.9997)
        add_tensor_38: "f32[72]" = torch.ops.aten.add.Tensor(mul_tensor_63, mul_tensor_64);  mul_tensor_63 = mul_tensor_64 = None
        add_tensor_39: "i64[]" = torch.ops.aten.add.Tensor(arg80_1, 1)
        squeeze_dims_39: "f32[72]" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3]);  getitem_27 = None
        squeeze_dims_40: "f32[72]" = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2, 3]);  rsqrt_13 = None
        mul_tensor_65: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_dims_39, 0.00029999999999996696)
        mul_tensor_66: "f32[72]" = torch.ops.aten.mul.Tensor(arg81_1, 0.9997)
        add_tensor_40: "f32[72]" = torch.ops.aten.add.Tensor(mul_tensor_65, mul_tensor_66);  mul_tensor_65 = mul_tensor_66 = None
        squeeze_dims_41: "f32[72]" = torch.ops.aten.squeeze.dims(getitem_26, [0, 2, 3]);  getitem_26 = None
        mul_tensor_67: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_dims_41, 1.0000398612827361);  squeeze_dims_41 = None
        mul_tensor_68: "f32[72]" = torch.ops.aten.mul.Tensor(mul_tensor_67, 0.00029999999999996696);  mul_tensor_67 = None
        mul_tensor_69: "f32[72]" = torch.ops.aten.mul.Tensor(arg82_1, 0.9997)
        add_tensor_41: "f32[72]" = torch.ops.aten.add.Tensor(mul_tensor_68, mul_tensor_69);  mul_tensor_68 = mul_tensor_69 = None
        add_tensor_42: "i64[]" = torch.ops.aten.add.Tensor(arg86_1, 1)
        squeeze_dims_42: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_29, [0, 2, 3]);  getitem_29 = None
        squeeze_dims_43: "f32[40]" = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2, 3]);  rsqrt_14 = None
        mul_tensor_70: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_dims_42, 0.00029999999999996696)
        mul_tensor_71: "f32[40]" = torch.ops.aten.mul.Tensor(arg87_1, 0.9997)
        add_tensor_43: "f32[40]" = torch.ops.aten.add.Tensor(mul_tensor_70, mul_tensor_71);  mul_tensor_70 = mul_tensor_71 = None
        squeeze_dims_44: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_28, [0, 2, 3]);  getitem_28 = None
        mul_tensor_72: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_dims_44, 1.0000398612827361);  squeeze_dims_44 = None
        mul_tensor_73: "f32[40]" = torch.ops.aten.mul.Tensor(mul_tensor_72, 0.00029999999999996696);  mul_tensor_72 = None
        mul_tensor_74: "f32[40]" = torch.ops.aten.mul.Tensor(arg88_1, 0.9997)
        add_tensor_44: "f32[40]" = torch.ops.aten.add.Tensor(mul_tensor_73, mul_tensor_74);  mul_tensor_73 = mul_tensor_74 = None
        add_tensor_45: "i64[]" = torch.ops.aten.add.Tensor(arg92_1, 1)
        squeeze_dims_45: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2, 3]);  getitem_31 = None
        squeeze_dims_46: "f32[120]" = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2, 3]);  rsqrt_15 = None
        mul_tensor_75: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_dims_45, 0.00029999999999996696)
        mul_tensor_76: "f32[120]" = torch.ops.aten.mul.Tensor(arg93_1, 0.9997)
        add_tensor_46: "f32[120]" = torch.ops.aten.add.Tensor(mul_tensor_75, mul_tensor_76);  mul_tensor_75 = mul_tensor_76 = None
        squeeze_dims_47: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_30, [0, 2, 3]);  getitem_30 = None
        mul_tensor_77: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_dims_47, 1.0000398612827361);  squeeze_dims_47 = None
        mul_tensor_78: "f32[120]" = torch.ops.aten.mul.Tensor(mul_tensor_77, 0.00029999999999996696);  mul_tensor_77 = None
        mul_tensor_79: "f32[120]" = torch.ops.aten.mul.Tensor(arg94_1, 0.9997)
        add_tensor_47: "f32[120]" = torch.ops.aten.add.Tensor(mul_tensor_78, mul_tensor_79);  mul_tensor_78 = mul_tensor_79 = None
        add_tensor_48: "i64[]" = torch.ops.aten.add.Tensor(arg98_1, 1)
        squeeze_dims_48: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3]);  getitem_33 = None
        squeeze_dims_49: "f32[120]" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2, 3]);  rsqrt_16 = None
        mul_tensor_80: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_dims_48, 0.00029999999999996696)
        mul_tensor_81: "f32[120]" = torch.ops.aten.mul.Tensor(arg99_1, 0.9997)
        add_tensor_49: "f32[120]" = torch.ops.aten.add.Tensor(mul_tensor_80, mul_tensor_81);  mul_tensor_80 = mul_tensor_81 = None
        squeeze_dims_50: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_32, [0, 2, 3]);  getitem_32 = None
        mul_tensor_82: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_dims_50, 1.0000398612827361);  squeeze_dims_50 = None
        mul_tensor_83: "f32[120]" = torch.ops.aten.mul.Tensor(mul_tensor_82, 0.00029999999999996696);  mul_tensor_82 = None
        mul_tensor_84: "f32[120]" = torch.ops.aten.mul.Tensor(arg100_1, 0.9997)
        add_tensor_50: "f32[120]" = torch.ops.aten.add.Tensor(mul_tensor_83, mul_tensor_84);  mul_tensor_83 = mul_tensor_84 = None
        add_tensor_51: "i64[]" = torch.ops.aten.add.Tensor(arg104_1, 1)
        squeeze_dims_51: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_35, [0, 2, 3]);  getitem_35 = None
        squeeze_dims_52: "f32[40]" = torch.ops.aten.squeeze.dims(rsqrt_17, [0, 2, 3]);  rsqrt_17 = None
        mul_tensor_85: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_dims_51, 0.00029999999999996696)
        mul_tensor_86: "f32[40]" = torch.ops.aten.mul.Tensor(arg105_1, 0.9997)
        add_tensor_52: "f32[40]" = torch.ops.aten.add.Tensor(mul_tensor_85, mul_tensor_86);  mul_tensor_85 = mul_tensor_86 = None
        squeeze_dims_53: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_34, [0, 2, 3]);  getitem_34 = None
        mul_tensor_87: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_dims_53, 1.0000398612827361);  squeeze_dims_53 = None
        mul_tensor_88: "f32[40]" = torch.ops.aten.mul.Tensor(mul_tensor_87, 0.00029999999999996696);  mul_tensor_87 = None
        mul_tensor_89: "f32[40]" = torch.ops.aten.mul.Tensor(arg106_1, 0.9997)
        add_tensor_53: "f32[40]" = torch.ops.aten.add.Tensor(mul_tensor_88, mul_tensor_89);  mul_tensor_88 = mul_tensor_89 = None
        add_tensor_54: "i64[]" = torch.ops.aten.add.Tensor(arg110_1, 1)
        squeeze_dims_54: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_37, [0, 2, 3]);  getitem_37 = None
        squeeze_dims_55: "f32[120]" = torch.ops.aten.squeeze.dims(rsqrt_18, [0, 2, 3]);  rsqrt_18 = None
        mul_tensor_90: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_dims_54, 0.00029999999999996696)
        mul_tensor_91: "f32[120]" = torch.ops.aten.mul.Tensor(arg111_1, 0.9997)
        add_tensor_55: "f32[120]" = torch.ops.aten.add.Tensor(mul_tensor_90, mul_tensor_91);  mul_tensor_90 = mul_tensor_91 = None
        squeeze_dims_56: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_36, [0, 2, 3]);  getitem_36 = None
        mul_tensor_92: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_dims_56, 1.0000398612827361);  squeeze_dims_56 = None
        mul_tensor_93: "f32[120]" = torch.ops.aten.mul.Tensor(mul_tensor_92, 0.00029999999999996696);  mul_tensor_92 = None
        mul_tensor_94: "f32[120]" = torch.ops.aten.mul.Tensor(arg112_1, 0.9997)
        add_tensor_56: "f32[120]" = torch.ops.aten.add.Tensor(mul_tensor_93, mul_tensor_94);  mul_tensor_93 = mul_tensor_94 = None
        add_tensor_57: "i64[]" = torch.ops.aten.add.Tensor(arg116_1, 1)
        squeeze_dims_57: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_39, [0, 2, 3]);  getitem_39 = None
        squeeze_dims_58: "f32[120]" = torch.ops.aten.squeeze.dims(rsqrt_19, [0, 2, 3]);  rsqrt_19 = None
        mul_tensor_95: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_dims_57, 0.00029999999999996696)
        mul_tensor_96: "f32[120]" = torch.ops.aten.mul.Tensor(arg117_1, 0.9997)
        add_tensor_58: "f32[120]" = torch.ops.aten.add.Tensor(mul_tensor_95, mul_tensor_96);  mul_tensor_95 = mul_tensor_96 = None
        squeeze_dims_59: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_38, [0, 2, 3]);  getitem_38 = None
        mul_tensor_97: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_dims_59, 1.0000398612827361);  squeeze_dims_59 = None
        mul_tensor_98: "f32[120]" = torch.ops.aten.mul.Tensor(mul_tensor_97, 0.00029999999999996696);  mul_tensor_97 = None
        mul_tensor_99: "f32[120]" = torch.ops.aten.mul.Tensor(arg118_1, 0.9997)
        add_tensor_59: "f32[120]" = torch.ops.aten.add.Tensor(mul_tensor_98, mul_tensor_99);  mul_tensor_98 = mul_tensor_99 = None
        add_tensor_60: "i64[]" = torch.ops.aten.add.Tensor(arg122_1, 1)
        squeeze_dims_60: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_41, [0, 2, 3]);  getitem_41 = None
        squeeze_dims_61: "f32[40]" = torch.ops.aten.squeeze.dims(rsqrt_20, [0, 2, 3]);  rsqrt_20 = None
        mul_tensor_100: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_dims_60, 0.00029999999999996696)
        mul_tensor_101: "f32[40]" = torch.ops.aten.mul.Tensor(arg123_1, 0.9997)
        add_tensor_61: "f32[40]" = torch.ops.aten.add.Tensor(mul_tensor_100, mul_tensor_101);  mul_tensor_100 = mul_tensor_101 = None
        squeeze_dims_62: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_40, [0, 2, 3]);  getitem_40 = None
        mul_tensor_102: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_dims_62, 1.0000398612827361);  squeeze_dims_62 = None
        mul_tensor_103: "f32[40]" = torch.ops.aten.mul.Tensor(mul_tensor_102, 0.00029999999999996696);  mul_tensor_102 = None
        mul_tensor_104: "f32[40]" = torch.ops.aten.mul.Tensor(arg124_1, 0.9997)
        add_tensor_62: "f32[40]" = torch.ops.aten.add.Tensor(mul_tensor_103, mul_tensor_104);  mul_tensor_103 = mul_tensor_104 = None
        add_tensor_63: "i64[]" = torch.ops.aten.add.Tensor(arg128_1, 1)
        squeeze_dims_63: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_43, [0, 2, 3]);  getitem_43 = None
        squeeze_dims_64: "f32[240]" = torch.ops.aten.squeeze.dims(rsqrt_21, [0, 2, 3]);  rsqrt_21 = None
        mul_tensor_105: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_dims_63, 0.00029999999999996696)
        mul_tensor_106: "f32[240]" = torch.ops.aten.mul.Tensor(arg129_1, 0.9997)
        add_tensor_64: "f32[240]" = torch.ops.aten.add.Tensor(mul_tensor_105, mul_tensor_106);  mul_tensor_105 = mul_tensor_106 = None
        squeeze_dims_65: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_42, [0, 2, 3]);  getitem_42 = None
        mul_tensor_107: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_dims_65, 1.0000398612827361);  squeeze_dims_65 = None
        mul_tensor_108: "f32[240]" = torch.ops.aten.mul.Tensor(mul_tensor_107, 0.00029999999999996696);  mul_tensor_107 = None
        mul_tensor_109: "f32[240]" = torch.ops.aten.mul.Tensor(arg130_1, 0.9997)
        add_tensor_65: "f32[240]" = torch.ops.aten.add.Tensor(mul_tensor_108, mul_tensor_109);  mul_tensor_108 = mul_tensor_109 = None
        add_tensor_66: "i64[]" = torch.ops.aten.add.Tensor(arg134_1, 1)
        squeeze_dims_66: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_45, [0, 2, 3]);  getitem_45 = None
        squeeze_dims_67: "f32[240]" = torch.ops.aten.squeeze.dims(rsqrt_22, [0, 2, 3]);  rsqrt_22 = None
        mul_tensor_110: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_dims_66, 0.00029999999999996696)
        mul_tensor_111: "f32[240]" = torch.ops.aten.mul.Tensor(arg135_1, 0.9997)
        add_tensor_67: "f32[240]" = torch.ops.aten.add.Tensor(mul_tensor_110, mul_tensor_111);  mul_tensor_110 = mul_tensor_111 = None
        squeeze_dims_68: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_44, [0, 2, 3]);  getitem_44 = None
        mul_tensor_112: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_dims_68, 1.0001594642002871);  squeeze_dims_68 = None
        mul_tensor_113: "f32[240]" = torch.ops.aten.mul.Tensor(mul_tensor_112, 0.00029999999999996696);  mul_tensor_112 = None
        mul_tensor_114: "f32[240]" = torch.ops.aten.mul.Tensor(arg136_1, 0.9997)
        add_tensor_68: "f32[240]" = torch.ops.aten.add.Tensor(mul_tensor_113, mul_tensor_114);  mul_tensor_113 = mul_tensor_114 = None
        add_tensor_69: "i64[]" = torch.ops.aten.add.Tensor(arg140_1, 1)
        squeeze_dims_69: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_47, [0, 2, 3]);  getitem_47 = None
        squeeze_dims_70: "f32[80]" = torch.ops.aten.squeeze.dims(rsqrt_23, [0, 2, 3]);  rsqrt_23 = None
        mul_tensor_115: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_dims_69, 0.00029999999999996696)
        mul_tensor_116: "f32[80]" = torch.ops.aten.mul.Tensor(arg141_1, 0.9997)
        add_tensor_70: "f32[80]" = torch.ops.aten.add.Tensor(mul_tensor_115, mul_tensor_116);  mul_tensor_115 = mul_tensor_116 = None
        squeeze_dims_71: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_46, [0, 2, 3]);  getitem_46 = None
        mul_tensor_117: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_dims_71, 1.0001594642002871);  squeeze_dims_71 = None
        mul_tensor_118: "f32[80]" = torch.ops.aten.mul.Tensor(mul_tensor_117, 0.00029999999999996696);  mul_tensor_117 = None
        mul_tensor_119: "f32[80]" = torch.ops.aten.mul.Tensor(arg142_1, 0.9997)
        add_tensor_71: "f32[80]" = torch.ops.aten.add.Tensor(mul_tensor_118, mul_tensor_119);  mul_tensor_118 = mul_tensor_119 = None
        add_tensor_72: "i64[]" = torch.ops.aten.add.Tensor(arg146_1, 1)
        squeeze_dims_72: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_49, [0, 2, 3]);  getitem_49 = None
        squeeze_dims_73: "f32[480]" = torch.ops.aten.squeeze.dims(rsqrt_24, [0, 2, 3]);  rsqrt_24 = None
        mul_tensor_120: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_dims_72, 0.00029999999999996696)
        mul_tensor_121: "f32[480]" = torch.ops.aten.mul.Tensor(arg147_1, 0.9997)
        add_tensor_73: "f32[480]" = torch.ops.aten.add.Tensor(mul_tensor_120, mul_tensor_121);  mul_tensor_120 = mul_tensor_121 = None
        squeeze_dims_74: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_48, [0, 2, 3]);  getitem_48 = None
        mul_tensor_122: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_dims_74, 1.0001594642002871);  squeeze_dims_74 = None
        mul_tensor_123: "f32[480]" = torch.ops.aten.mul.Tensor(mul_tensor_122, 0.00029999999999996696);  mul_tensor_122 = None
        mul_tensor_124: "f32[480]" = torch.ops.aten.mul.Tensor(arg148_1, 0.9997)
        add_tensor_74: "f32[480]" = torch.ops.aten.add.Tensor(mul_tensor_123, mul_tensor_124);  mul_tensor_123 = mul_tensor_124 = None
        add_tensor_75: "i64[]" = torch.ops.aten.add.Tensor(arg152_1, 1)
        squeeze_dims_75: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_51, [0, 2, 3]);  getitem_51 = None
        squeeze_dims_76: "f32[480]" = torch.ops.aten.squeeze.dims(rsqrt_25, [0, 2, 3]);  rsqrt_25 = None
        mul_tensor_125: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_dims_75, 0.00029999999999996696)
        mul_tensor_126: "f32[480]" = torch.ops.aten.mul.Tensor(arg153_1, 0.9997)
        add_tensor_76: "f32[480]" = torch.ops.aten.add.Tensor(mul_tensor_125, mul_tensor_126);  mul_tensor_125 = mul_tensor_126 = None
        squeeze_dims_77: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_50, [0, 2, 3]);  getitem_50 = None
        mul_tensor_127: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_dims_77, 1.0001594642002871);  squeeze_dims_77 = None
        mul_tensor_128: "f32[480]" = torch.ops.aten.mul.Tensor(mul_tensor_127, 0.00029999999999996696);  mul_tensor_127 = None
        mul_tensor_129: "f32[480]" = torch.ops.aten.mul.Tensor(arg154_1, 0.9997)
        add_tensor_77: "f32[480]" = torch.ops.aten.add.Tensor(mul_tensor_128, mul_tensor_129);  mul_tensor_128 = mul_tensor_129 = None
        add_tensor_78: "i64[]" = torch.ops.aten.add.Tensor(arg158_1, 1)
        squeeze_dims_78: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_53, [0, 2, 3]);  getitem_53 = None
        squeeze_dims_79: "f32[80]" = torch.ops.aten.squeeze.dims(rsqrt_26, [0, 2, 3]);  rsqrt_26 = None
        mul_tensor_130: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_dims_78, 0.00029999999999996696)
        mul_tensor_131: "f32[80]" = torch.ops.aten.mul.Tensor(arg159_1, 0.9997)
        add_tensor_79: "f32[80]" = torch.ops.aten.add.Tensor(mul_tensor_130, mul_tensor_131);  mul_tensor_130 = mul_tensor_131 = None
        squeeze_dims_80: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_52, [0, 2, 3]);  getitem_52 = None
        mul_tensor_132: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_dims_80, 1.0001594642002871);  squeeze_dims_80 = None
        mul_tensor_133: "f32[80]" = torch.ops.aten.mul.Tensor(mul_tensor_132, 0.00029999999999996696);  mul_tensor_132 = None
        mul_tensor_134: "f32[80]" = torch.ops.aten.mul.Tensor(arg160_1, 0.9997)
        add_tensor_80: "f32[80]" = torch.ops.aten.add.Tensor(mul_tensor_133, mul_tensor_134);  mul_tensor_133 = mul_tensor_134 = None
        add_tensor_81: "i64[]" = torch.ops.aten.add.Tensor(arg164_1, 1)
        squeeze_dims_81: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_55, [0, 2, 3]);  getitem_55 = None
        squeeze_dims_82: "f32[480]" = torch.ops.aten.squeeze.dims(rsqrt_27, [0, 2, 3]);  rsqrt_27 = None
        mul_tensor_135: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_dims_81, 0.00029999999999996696)
        mul_tensor_136: "f32[480]" = torch.ops.aten.mul.Tensor(arg165_1, 0.9997)
        add_tensor_82: "f32[480]" = torch.ops.aten.add.Tensor(mul_tensor_135, mul_tensor_136);  mul_tensor_135 = mul_tensor_136 = None
        squeeze_dims_83: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_54, [0, 2, 3]);  getitem_54 = None
        mul_tensor_137: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_dims_83, 1.0001594642002871);  squeeze_dims_83 = None
        mul_tensor_138: "f32[480]" = torch.ops.aten.mul.Tensor(mul_tensor_137, 0.00029999999999996696);  mul_tensor_137 = None
        mul_tensor_139: "f32[480]" = torch.ops.aten.mul.Tensor(arg166_1, 0.9997)
        add_tensor_83: "f32[480]" = torch.ops.aten.add.Tensor(mul_tensor_138, mul_tensor_139);  mul_tensor_138 = mul_tensor_139 = None
        add_tensor_84: "i64[]" = torch.ops.aten.add.Tensor(arg170_1, 1)
        squeeze_dims_84: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_57, [0, 2, 3]);  getitem_57 = None
        squeeze_dims_85: "f32[480]" = torch.ops.aten.squeeze.dims(rsqrt_28, [0, 2, 3]);  rsqrt_28 = None
        mul_tensor_140: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_dims_84, 0.00029999999999996696)
        mul_tensor_141: "f32[480]" = torch.ops.aten.mul.Tensor(arg171_1, 0.9997)
        add_tensor_85: "f32[480]" = torch.ops.aten.add.Tensor(mul_tensor_140, mul_tensor_141);  mul_tensor_140 = mul_tensor_141 = None
        squeeze_dims_86: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_56, [0, 2, 3]);  getitem_56 = None
        mul_tensor_142: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_dims_86, 1.0001594642002871);  squeeze_dims_86 = None
        mul_tensor_143: "f32[480]" = torch.ops.aten.mul.Tensor(mul_tensor_142, 0.00029999999999996696);  mul_tensor_142 = None
        mul_tensor_144: "f32[480]" = torch.ops.aten.mul.Tensor(arg172_1, 0.9997)
        add_tensor_86: "f32[480]" = torch.ops.aten.add.Tensor(mul_tensor_143, mul_tensor_144);  mul_tensor_143 = mul_tensor_144 = None
        add_tensor_87: "i64[]" = torch.ops.aten.add.Tensor(arg176_1, 1)
        squeeze_dims_87: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_59, [0, 2, 3]);  getitem_59 = None
        squeeze_dims_88: "f32[80]" = torch.ops.aten.squeeze.dims(rsqrt_29, [0, 2, 3]);  rsqrt_29 = None
        mul_tensor_145: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_dims_87, 0.00029999999999996696)
        mul_tensor_146: "f32[80]" = torch.ops.aten.mul.Tensor(arg177_1, 0.9997)
        add_tensor_88: "f32[80]" = torch.ops.aten.add.Tensor(mul_tensor_145, mul_tensor_146);  mul_tensor_145 = mul_tensor_146 = None
        squeeze_dims_89: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_58, [0, 2, 3]);  getitem_58 = None
        mul_tensor_147: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_dims_89, 1.0001594642002871);  squeeze_dims_89 = None
        mul_tensor_148: "f32[80]" = torch.ops.aten.mul.Tensor(mul_tensor_147, 0.00029999999999996696);  mul_tensor_147 = None
        mul_tensor_149: "f32[80]" = torch.ops.aten.mul.Tensor(arg178_1, 0.9997)
        add_tensor_89: "f32[80]" = torch.ops.aten.add.Tensor(mul_tensor_148, mul_tensor_149);  mul_tensor_148 = mul_tensor_149 = None
        add_tensor_90: "i64[]" = torch.ops.aten.add.Tensor(arg182_1, 1)
        squeeze_dims_90: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_61, [0, 2, 3]);  getitem_61 = None
        squeeze_dims_91: "f32[480]" = torch.ops.aten.squeeze.dims(rsqrt_30, [0, 2, 3]);  rsqrt_30 = None
        mul_tensor_150: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_dims_90, 0.00029999999999996696)
        mul_tensor_151: "f32[480]" = torch.ops.aten.mul.Tensor(arg183_1, 0.9997)
        add_tensor_91: "f32[480]" = torch.ops.aten.add.Tensor(mul_tensor_150, mul_tensor_151);  mul_tensor_150 = mul_tensor_151 = None
        squeeze_dims_92: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_60, [0, 2, 3]);  getitem_60 = None
        mul_tensor_152: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_dims_92, 1.0001594642002871);  squeeze_dims_92 = None
        mul_tensor_153: "f32[480]" = torch.ops.aten.mul.Tensor(mul_tensor_152, 0.00029999999999996696);  mul_tensor_152 = None
        mul_tensor_154: "f32[480]" = torch.ops.aten.mul.Tensor(arg184_1, 0.9997)
        add_tensor_92: "f32[480]" = torch.ops.aten.add.Tensor(mul_tensor_153, mul_tensor_154);  mul_tensor_153 = mul_tensor_154 = None
        add_tensor_93: "i64[]" = torch.ops.aten.add.Tensor(arg188_1, 1)
        squeeze_dims_93: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_63, [0, 2, 3]);  getitem_63 = None
        squeeze_dims_94: "f32[480]" = torch.ops.aten.squeeze.dims(rsqrt_31, [0, 2, 3]);  rsqrt_31 = None
        mul_tensor_155: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_dims_93, 0.00029999999999996696)
        mul_tensor_156: "f32[480]" = torch.ops.aten.mul.Tensor(arg189_1, 0.9997)
        add_tensor_94: "f32[480]" = torch.ops.aten.add.Tensor(mul_tensor_155, mul_tensor_156);  mul_tensor_155 = mul_tensor_156 = None
        squeeze_dims_95: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_62, [0, 2, 3]);  getitem_62 = None
        mul_tensor_157: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_dims_95, 1.0001594642002871);  squeeze_dims_95 = None
        mul_tensor_158: "f32[480]" = torch.ops.aten.mul.Tensor(mul_tensor_157, 0.00029999999999996696);  mul_tensor_157 = None
        mul_tensor_159: "f32[480]" = torch.ops.aten.mul.Tensor(arg190_1, 0.9997)
        add_tensor_95: "f32[480]" = torch.ops.aten.add.Tensor(mul_tensor_158, mul_tensor_159);  mul_tensor_158 = mul_tensor_159 = None
        add_tensor_96: "i64[]" = torch.ops.aten.add.Tensor(arg194_1, 1)
        squeeze_dims_96: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_65, [0, 2, 3]);  getitem_65 = None
        squeeze_dims_97: "f32[96]" = torch.ops.aten.squeeze.dims(rsqrt_32, [0, 2, 3]);  rsqrt_32 = None
        mul_tensor_160: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_dims_96, 0.00029999999999996696)
        mul_tensor_161: "f32[96]" = torch.ops.aten.mul.Tensor(arg195_1, 0.9997)
        add_tensor_97: "f32[96]" = torch.ops.aten.add.Tensor(mul_tensor_160, mul_tensor_161);  mul_tensor_160 = mul_tensor_161 = None
        squeeze_dims_98: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_64, [0, 2, 3]);  getitem_64 = None
        mul_tensor_162: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_dims_98, 1.0001594642002871);  squeeze_dims_98 = None
        mul_tensor_163: "f32[96]" = torch.ops.aten.mul.Tensor(mul_tensor_162, 0.00029999999999996696);  mul_tensor_162 = None
        mul_tensor_164: "f32[96]" = torch.ops.aten.mul.Tensor(arg196_1, 0.9997)
        add_tensor_98: "f32[96]" = torch.ops.aten.add.Tensor(mul_tensor_163, mul_tensor_164);  mul_tensor_163 = mul_tensor_164 = None
        add_tensor_99: "i64[]" = torch.ops.aten.add.Tensor(arg200_1, 1)
        squeeze_dims_99: "f32[576]" = torch.ops.aten.squeeze.dims(getitem_67, [0, 2, 3]);  getitem_67 = None
        squeeze_dims_100: "f32[576]" = torch.ops.aten.squeeze.dims(rsqrt_33, [0, 2, 3]);  rsqrt_33 = None
        mul_tensor_165: "f32[576]" = torch.ops.aten.mul.Tensor(squeeze_dims_99, 0.00029999999999996696)
        mul_tensor_166: "f32[576]" = torch.ops.aten.mul.Tensor(arg201_1, 0.9997)
        add_tensor_100: "f32[576]" = torch.ops.aten.add.Tensor(mul_tensor_165, mul_tensor_166);  mul_tensor_165 = mul_tensor_166 = None
        squeeze_dims_101: "f32[576]" = torch.ops.aten.squeeze.dims(getitem_66, [0, 2, 3]);  getitem_66 = None
        mul_tensor_167: "f32[576]" = torch.ops.aten.mul.Tensor(squeeze_dims_101, 1.0001594642002871);  squeeze_dims_101 = None
        mul_tensor_168: "f32[576]" = torch.ops.aten.mul.Tensor(mul_tensor_167, 0.00029999999999996696);  mul_tensor_167 = None
        mul_tensor_169: "f32[576]" = torch.ops.aten.mul.Tensor(arg202_1, 0.9997)
        add_tensor_101: "f32[576]" = torch.ops.aten.add.Tensor(mul_tensor_168, mul_tensor_169);  mul_tensor_168 = mul_tensor_169 = None
        add_tensor_102: "i64[]" = torch.ops.aten.add.Tensor(arg206_1, 1)
        squeeze_dims_102: "f32[576]" = torch.ops.aten.squeeze.dims(getitem_69, [0, 2, 3]);  getitem_69 = None
        squeeze_dims_103: "f32[576]" = torch.ops.aten.squeeze.dims(rsqrt_34, [0, 2, 3]);  rsqrt_34 = None
        mul_tensor_170: "f32[576]" = torch.ops.aten.mul.Tensor(squeeze_dims_102, 0.00029999999999996696)
        mul_tensor_171: "f32[576]" = torch.ops.aten.mul.Tensor(arg207_1, 0.9997)
        add_tensor_103: "f32[576]" = torch.ops.aten.add.Tensor(mul_tensor_170, mul_tensor_171);  mul_tensor_170 = mul_tensor_171 = None
        squeeze_dims_104: "f32[576]" = torch.ops.aten.squeeze.dims(getitem_68, [0, 2, 3]);  getitem_68 = None
        mul_tensor_172: "f32[576]" = torch.ops.aten.mul.Tensor(squeeze_dims_104, 1.0001594642002871);  squeeze_dims_104 = None
        mul_tensor_173: "f32[576]" = torch.ops.aten.mul.Tensor(mul_tensor_172, 0.00029999999999996696);  mul_tensor_172 = None
        mul_tensor_174: "f32[576]" = torch.ops.aten.mul.Tensor(arg208_1, 0.9997)
        add_tensor_104: "f32[576]" = torch.ops.aten.add.Tensor(mul_tensor_173, mul_tensor_174);  mul_tensor_173 = mul_tensor_174 = None
        add_tensor_105: "i64[]" = torch.ops.aten.add.Tensor(arg212_1, 1)
        squeeze_dims_105: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_71, [0, 2, 3]);  getitem_71 = None
        squeeze_dims_106: "f32[96]" = torch.ops.aten.squeeze.dims(rsqrt_35, [0, 2, 3]);  rsqrt_35 = None
        mul_tensor_175: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_dims_105, 0.00029999999999996696)
        mul_tensor_176: "f32[96]" = torch.ops.aten.mul.Tensor(arg213_1, 0.9997)
        add_tensor_106: "f32[96]" = torch.ops.aten.add.Tensor(mul_tensor_175, mul_tensor_176);  mul_tensor_175 = mul_tensor_176 = None
        squeeze_dims_107: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_70, [0, 2, 3]);  getitem_70 = None
        mul_tensor_177: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_dims_107, 1.0001594642002871);  squeeze_dims_107 = None
        mul_tensor_178: "f32[96]" = torch.ops.aten.mul.Tensor(mul_tensor_177, 0.00029999999999996696);  mul_tensor_177 = None
        mul_tensor_179: "f32[96]" = torch.ops.aten.mul.Tensor(arg214_1, 0.9997)
        add_tensor_107: "f32[96]" = torch.ops.aten.add.Tensor(mul_tensor_178, mul_tensor_179);  mul_tensor_178 = mul_tensor_179 = None
        add_tensor_108: "i64[]" = torch.ops.aten.add.Tensor(arg218_1, 1)
        squeeze_dims_108: "f32[576]" = torch.ops.aten.squeeze.dims(getitem_73, [0, 2, 3]);  getitem_73 = None
        squeeze_dims_109: "f32[576]" = torch.ops.aten.squeeze.dims(rsqrt_36, [0, 2, 3]);  rsqrt_36 = None
        mul_tensor_180: "f32[576]" = torch.ops.aten.mul.Tensor(squeeze_dims_108, 0.00029999999999996696)
        mul_tensor_181: "f32[576]" = torch.ops.aten.mul.Tensor(arg219_1, 0.9997)
        add_tensor_109: "f32[576]" = torch.ops.aten.add.Tensor(mul_tensor_180, mul_tensor_181);  mul_tensor_180 = mul_tensor_181 = None
        squeeze_dims_110: "f32[576]" = torch.ops.aten.squeeze.dims(getitem_72, [0, 2, 3]);  getitem_72 = None
        mul_tensor_182: "f32[576]" = torch.ops.aten.mul.Tensor(squeeze_dims_110, 1.0001594642002871);  squeeze_dims_110 = None
        mul_tensor_183: "f32[576]" = torch.ops.aten.mul.Tensor(mul_tensor_182, 0.00029999999999996696);  mul_tensor_182 = None
        mul_tensor_184: "f32[576]" = torch.ops.aten.mul.Tensor(arg220_1, 0.9997)
        add_tensor_110: "f32[576]" = torch.ops.aten.add.Tensor(mul_tensor_183, mul_tensor_184);  mul_tensor_183 = mul_tensor_184 = None
        add_tensor_111: "i64[]" = torch.ops.aten.add.Tensor(arg224_1, 1)
        squeeze_dims_111: "f32[576]" = torch.ops.aten.squeeze.dims(getitem_75, [0, 2, 3]);  getitem_75 = None
        squeeze_dims_112: "f32[576]" = torch.ops.aten.squeeze.dims(rsqrt_37, [0, 2, 3]);  rsqrt_37 = None
        mul_tensor_185: "f32[576]" = torch.ops.aten.mul.Tensor(squeeze_dims_111, 0.00029999999999996696)
        mul_tensor_186: "f32[576]" = torch.ops.aten.mul.Tensor(arg225_1, 0.9997)
        add_tensor_112: "f32[576]" = torch.ops.aten.add.Tensor(mul_tensor_185, mul_tensor_186);  mul_tensor_185 = mul_tensor_186 = None
        squeeze_dims_113: "f32[576]" = torch.ops.aten.squeeze.dims(getitem_74, [0, 2, 3]);  getitem_74 = None
        mul_tensor_187: "f32[576]" = torch.ops.aten.mul.Tensor(squeeze_dims_113, 1.0006381620931717);  squeeze_dims_113 = None
        mul_tensor_188: "f32[576]" = torch.ops.aten.mul.Tensor(mul_tensor_187, 0.00029999999999996696);  mul_tensor_187 = None
        mul_tensor_189: "f32[576]" = torch.ops.aten.mul.Tensor(arg226_1, 0.9997)
        add_tensor_113: "f32[576]" = torch.ops.aten.add.Tensor(mul_tensor_188, mul_tensor_189);  mul_tensor_188 = mul_tensor_189 = None
        add_tensor_114: "i64[]" = torch.ops.aten.add.Tensor(arg230_1, 1)
        squeeze_dims_114: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_77, [0, 2, 3]);  getitem_77 = None
        squeeze_dims_115: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_38, [0, 2, 3]);  rsqrt_38 = None
        mul_tensor_190: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_114, 0.00029999999999996696)
        mul_tensor_191: "f32[192]" = torch.ops.aten.mul.Tensor(arg231_1, 0.9997)
        add_tensor_115: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_190, mul_tensor_191);  mul_tensor_190 = mul_tensor_191 = None
        squeeze_dims_116: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_76, [0, 2, 3]);  getitem_76 = None
        mul_tensor_192: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_116, 1.0006381620931717);  squeeze_dims_116 = None
        mul_tensor_193: "f32[192]" = torch.ops.aten.mul.Tensor(mul_tensor_192, 0.00029999999999996696);  mul_tensor_192 = None
        mul_tensor_194: "f32[192]" = torch.ops.aten.mul.Tensor(arg232_1, 0.9997)
        add_tensor_116: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_193, mul_tensor_194);  mul_tensor_193 = mul_tensor_194 = None
        add_tensor_117: "i64[]" = torch.ops.aten.add.Tensor(arg236_1, 1)
        squeeze_dims_117: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_79, [0, 2, 3]);  getitem_79 = None
        squeeze_dims_118: "f32[1152]" = torch.ops.aten.squeeze.dims(rsqrt_39, [0, 2, 3]);  rsqrt_39 = None
        mul_tensor_195: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_dims_117, 0.00029999999999996696)
        mul_tensor_196: "f32[1152]" = torch.ops.aten.mul.Tensor(arg237_1, 0.9997)
        add_tensor_118: "f32[1152]" = torch.ops.aten.add.Tensor(mul_tensor_195, mul_tensor_196);  mul_tensor_195 = mul_tensor_196 = None
        squeeze_dims_119: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_78, [0, 2, 3]);  getitem_78 = None
        mul_tensor_197: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_dims_119, 1.0006381620931717);  squeeze_dims_119 = None
        mul_tensor_198: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_tensor_197, 0.00029999999999996696);  mul_tensor_197 = None
        mul_tensor_199: "f32[1152]" = torch.ops.aten.mul.Tensor(arg238_1, 0.9997)
        add_tensor_119: "f32[1152]" = torch.ops.aten.add.Tensor(mul_tensor_198, mul_tensor_199);  mul_tensor_198 = mul_tensor_199 = None
        add_tensor_120: "i64[]" = torch.ops.aten.add.Tensor(arg242_1, 1)
        squeeze_dims_120: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_81, [0, 2, 3]);  getitem_81 = None
        squeeze_dims_121: "f32[1152]" = torch.ops.aten.squeeze.dims(rsqrt_40, [0, 2, 3]);  rsqrt_40 = None
        mul_tensor_200: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_dims_120, 0.00029999999999996696)
        mul_tensor_201: "f32[1152]" = torch.ops.aten.mul.Tensor(arg243_1, 0.9997)
        add_tensor_121: "f32[1152]" = torch.ops.aten.add.Tensor(mul_tensor_200, mul_tensor_201);  mul_tensor_200 = mul_tensor_201 = None
        squeeze_dims_122: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_80, [0, 2, 3]);  getitem_80 = None
        mul_tensor_202: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_dims_122, 1.0006381620931717);  squeeze_dims_122 = None
        mul_tensor_203: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_tensor_202, 0.00029999999999996696);  mul_tensor_202 = None
        mul_tensor_204: "f32[1152]" = torch.ops.aten.mul.Tensor(arg244_1, 0.9997)
        add_tensor_122: "f32[1152]" = torch.ops.aten.add.Tensor(mul_tensor_203, mul_tensor_204);  mul_tensor_203 = mul_tensor_204 = None
        add_tensor_123: "i64[]" = torch.ops.aten.add.Tensor(arg248_1, 1)
        squeeze_dims_123: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_83, [0, 2, 3]);  getitem_83 = None
        squeeze_dims_124: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_41, [0, 2, 3]);  rsqrt_41 = None
        mul_tensor_205: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_123, 0.00029999999999996696)
        mul_tensor_206: "f32[192]" = torch.ops.aten.mul.Tensor(arg249_1, 0.9997)
        add_tensor_124: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_205, mul_tensor_206);  mul_tensor_205 = mul_tensor_206 = None
        squeeze_dims_125: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_82, [0, 2, 3]);  getitem_82 = None
        mul_tensor_207: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_125, 1.0006381620931717);  squeeze_dims_125 = None
        mul_tensor_208: "f32[192]" = torch.ops.aten.mul.Tensor(mul_tensor_207, 0.00029999999999996696);  mul_tensor_207 = None
        mul_tensor_209: "f32[192]" = torch.ops.aten.mul.Tensor(arg250_1, 0.9997)
        add_tensor_125: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_208, mul_tensor_209);  mul_tensor_208 = mul_tensor_209 = None
        add_tensor_126: "i64[]" = torch.ops.aten.add.Tensor(arg254_1, 1)
        squeeze_dims_126: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_85, [0, 2, 3]);  getitem_85 = None
        squeeze_dims_127: "f32[1152]" = torch.ops.aten.squeeze.dims(rsqrt_42, [0, 2, 3]);  rsqrt_42 = None
        mul_tensor_210: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_dims_126, 0.00029999999999996696)
        mul_tensor_211: "f32[1152]" = torch.ops.aten.mul.Tensor(arg255_1, 0.9997)
        add_tensor_127: "f32[1152]" = torch.ops.aten.add.Tensor(mul_tensor_210, mul_tensor_211);  mul_tensor_210 = mul_tensor_211 = None
        squeeze_dims_128: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_84, [0, 2, 3]);  getitem_84 = None
        mul_tensor_212: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_dims_128, 1.0006381620931717);  squeeze_dims_128 = None
        mul_tensor_213: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_tensor_212, 0.00029999999999996696);  mul_tensor_212 = None
        mul_tensor_214: "f32[1152]" = torch.ops.aten.mul.Tensor(arg256_1, 0.9997)
        add_tensor_128: "f32[1152]" = torch.ops.aten.add.Tensor(mul_tensor_213, mul_tensor_214);  mul_tensor_213 = mul_tensor_214 = None
        add_tensor_129: "i64[]" = torch.ops.aten.add.Tensor(arg260_1, 1)
        squeeze_dims_129: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_87, [0, 2, 3]);  getitem_87 = None
        squeeze_dims_130: "f32[1152]" = torch.ops.aten.squeeze.dims(rsqrt_43, [0, 2, 3]);  rsqrt_43 = None
        mul_tensor_215: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_dims_129, 0.00029999999999996696)
        mul_tensor_216: "f32[1152]" = torch.ops.aten.mul.Tensor(arg261_1, 0.9997)
        add_tensor_130: "f32[1152]" = torch.ops.aten.add.Tensor(mul_tensor_215, mul_tensor_216);  mul_tensor_215 = mul_tensor_216 = None
        squeeze_dims_131: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_86, [0, 2, 3]);  getitem_86 = None
        mul_tensor_217: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_dims_131, 1.0006381620931717);  squeeze_dims_131 = None
        mul_tensor_218: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_tensor_217, 0.00029999999999996696);  mul_tensor_217 = None
        mul_tensor_219: "f32[1152]" = torch.ops.aten.mul.Tensor(arg262_1, 0.9997)
        add_tensor_131: "f32[1152]" = torch.ops.aten.add.Tensor(mul_tensor_218, mul_tensor_219);  mul_tensor_218 = mul_tensor_219 = None
        add_tensor_132: "i64[]" = torch.ops.aten.add.Tensor(arg266_1, 1)
        squeeze_dims_132: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_89, [0, 2, 3]);  getitem_89 = None
        squeeze_dims_133: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_44, [0, 2, 3]);  rsqrt_44 = None
        mul_tensor_220: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_132, 0.00029999999999996696)
        mul_tensor_221: "f32[192]" = torch.ops.aten.mul.Tensor(arg267_1, 0.9997)
        add_tensor_133: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_220, mul_tensor_221);  mul_tensor_220 = mul_tensor_221 = None
        squeeze_dims_134: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_88, [0, 2, 3]);  getitem_88 = None
        mul_tensor_222: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_134, 1.0006381620931717);  squeeze_dims_134 = None
        mul_tensor_223: "f32[192]" = torch.ops.aten.mul.Tensor(mul_tensor_222, 0.00029999999999996696);  mul_tensor_222 = None
        mul_tensor_224: "f32[192]" = torch.ops.aten.mul.Tensor(arg268_1, 0.9997)
        add_tensor_134: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_223, mul_tensor_224);  mul_tensor_223 = mul_tensor_224 = None
        add_tensor_135: "i64[]" = torch.ops.aten.add.Tensor(arg272_1, 1)
        squeeze_dims_135: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_91, [0, 2, 3]);  getitem_91 = None
        squeeze_dims_136: "f32[1152]" = torch.ops.aten.squeeze.dims(rsqrt_45, [0, 2, 3]);  rsqrt_45 = None
        mul_tensor_225: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_dims_135, 0.00029999999999996696)
        mul_tensor_226: "f32[1152]" = torch.ops.aten.mul.Tensor(arg273_1, 0.9997)
        add_tensor_136: "f32[1152]" = torch.ops.aten.add.Tensor(mul_tensor_225, mul_tensor_226);  mul_tensor_225 = mul_tensor_226 = None
        squeeze_dims_137: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_90, [0, 2, 3]);  getitem_90 = None
        mul_tensor_227: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_dims_137, 1.0006381620931717);  squeeze_dims_137 = None
        mul_tensor_228: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_tensor_227, 0.00029999999999996696);  mul_tensor_227 = None
        mul_tensor_229: "f32[1152]" = torch.ops.aten.mul.Tensor(arg274_1, 0.9997)
        add_tensor_137: "f32[1152]" = torch.ops.aten.add.Tensor(mul_tensor_228, mul_tensor_229);  mul_tensor_228 = mul_tensor_229 = None
        add_tensor_138: "i64[]" = torch.ops.aten.add.Tensor(arg278_1, 1)
        squeeze_dims_138: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_93, [0, 2, 3]);  getitem_93 = None
        squeeze_dims_139: "f32[1152]" = torch.ops.aten.squeeze.dims(rsqrt_46, [0, 2, 3]);  rsqrt_46 = None
        mul_tensor_230: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_dims_138, 0.00029999999999996696)
        mul_tensor_231: "f32[1152]" = torch.ops.aten.mul.Tensor(arg279_1, 0.9997)
        add_tensor_139: "f32[1152]" = torch.ops.aten.add.Tensor(mul_tensor_230, mul_tensor_231);  mul_tensor_230 = mul_tensor_231 = None
        squeeze_dims_140: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_92, [0, 2, 3]);  getitem_92 = None
        mul_tensor_232: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_dims_140, 1.0006381620931717);  squeeze_dims_140 = None
        mul_tensor_233: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_tensor_232, 0.00029999999999996696);  mul_tensor_232 = None
        mul_tensor_234: "f32[1152]" = torch.ops.aten.mul.Tensor(arg280_1, 0.9997)
        add_tensor_140: "f32[1152]" = torch.ops.aten.add.Tensor(mul_tensor_233, mul_tensor_234);  mul_tensor_233 = mul_tensor_234 = None
        add_tensor_141: "i64[]" = torch.ops.aten.add.Tensor(arg284_1, 1)
        squeeze_dims_141: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_95, [0, 2, 3]);  getitem_95 = None
        squeeze_dims_142: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_47, [0, 2, 3]);  rsqrt_47 = None
        mul_tensor_235: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_141, 0.00029999999999996696)
        mul_tensor_236: "f32[192]" = torch.ops.aten.mul.Tensor(arg285_1, 0.9997)
        add_tensor_142: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_235, mul_tensor_236);  mul_tensor_235 = mul_tensor_236 = None
        squeeze_dims_143: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_94, [0, 2, 3]);  getitem_94 = None
        mul_tensor_237: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_143, 1.0006381620931717);  squeeze_dims_143 = None
        mul_tensor_238: "f32[192]" = torch.ops.aten.mul.Tensor(mul_tensor_237, 0.00029999999999996696);  mul_tensor_237 = None
        mul_tensor_239: "f32[192]" = torch.ops.aten.mul.Tensor(arg286_1, 0.9997)
        add_tensor_143: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_238, mul_tensor_239);  mul_tensor_238 = mul_tensor_239 = None
        add_tensor_144: "i64[]" = torch.ops.aten.add.Tensor(arg290_1, 1)
        squeeze_dims_144: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_97, [0, 2, 3]);  getitem_97 = None
        squeeze_dims_145: "f32[1152]" = torch.ops.aten.squeeze.dims(rsqrt_48, [0, 2, 3]);  rsqrt_48 = None
        mul_tensor_240: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_dims_144, 0.00029999999999996696)
        mul_tensor_241: "f32[1152]" = torch.ops.aten.mul.Tensor(arg291_1, 0.9997)
        add_tensor_145: "f32[1152]" = torch.ops.aten.add.Tensor(mul_tensor_240, mul_tensor_241);  mul_tensor_240 = mul_tensor_241 = None
        squeeze_dims_146: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_96, [0, 2, 3]);  getitem_96 = None
        mul_tensor_242: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_dims_146, 1.0006381620931717);  squeeze_dims_146 = None
        mul_tensor_243: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_tensor_242, 0.00029999999999996696);  mul_tensor_242 = None
        mul_tensor_244: "f32[1152]" = torch.ops.aten.mul.Tensor(arg292_1, 0.9997)
        add_tensor_146: "f32[1152]" = torch.ops.aten.add.Tensor(mul_tensor_243, mul_tensor_244);  mul_tensor_243 = mul_tensor_244 = None
        add_tensor_147: "i64[]" = torch.ops.aten.add.Tensor(arg296_1, 1)
        squeeze_dims_147: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_99, [0, 2, 3]);  getitem_99 = None
        squeeze_dims_148: "f32[1152]" = torch.ops.aten.squeeze.dims(rsqrt_49, [0, 2, 3]);  rsqrt_49 = None
        mul_tensor_245: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_dims_147, 0.00029999999999996696)
        mul_tensor_246: "f32[1152]" = torch.ops.aten.mul.Tensor(arg297_1, 0.9997)
        add_tensor_148: "f32[1152]" = torch.ops.aten.add.Tensor(mul_tensor_245, mul_tensor_246);  mul_tensor_245 = mul_tensor_246 = None
        squeeze_dims_149: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_98, [0, 2, 3]);  getitem_98 = None
        mul_tensor_247: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_dims_149, 1.0006381620931717);  squeeze_dims_149 = None
        mul_tensor_248: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_tensor_247, 0.00029999999999996696);  mul_tensor_247 = None
        mul_tensor_249: "f32[1152]" = torch.ops.aten.mul.Tensor(arg298_1, 0.9997)
        add_tensor_149: "f32[1152]" = torch.ops.aten.add.Tensor(mul_tensor_248, mul_tensor_249);  mul_tensor_248 = mul_tensor_249 = None
        add_tensor_150: "i64[]" = torch.ops.aten.add.Tensor(arg302_1, 1)
        squeeze_dims_150: "f32[320]" = torch.ops.aten.squeeze.dims(getitem_101, [0, 2, 3]);  getitem_101 = None
        squeeze_dims_151: "f32[320]" = torch.ops.aten.squeeze.dims(rsqrt_50, [0, 2, 3]);  rsqrt_50 = None
        mul_tensor_250: "f32[320]" = torch.ops.aten.mul.Tensor(squeeze_dims_150, 0.00029999999999996696)
        mul_tensor_251: "f32[320]" = torch.ops.aten.mul.Tensor(arg303_1, 0.9997)
        add_tensor_151: "f32[320]" = torch.ops.aten.add.Tensor(mul_tensor_250, mul_tensor_251);  mul_tensor_250 = mul_tensor_251 = None
        squeeze_dims_152: "f32[320]" = torch.ops.aten.squeeze.dims(getitem_100, [0, 2, 3]);  getitem_100 = None
        mul_tensor_252: "f32[320]" = torch.ops.aten.mul.Tensor(squeeze_dims_152, 1.0006381620931717);  squeeze_dims_152 = None
        mul_tensor_253: "f32[320]" = torch.ops.aten.mul.Tensor(mul_tensor_252, 0.00029999999999996696);  mul_tensor_252 = None
        mul_tensor_254: "f32[320]" = torch.ops.aten.mul.Tensor(arg304_1, 0.9997)
        add_tensor_152: "f32[320]" = torch.ops.aten.add.Tensor(mul_tensor_253, mul_tensor_254);  mul_tensor_253 = mul_tensor_254 = None
        add_tensor_153: "i64[]" = torch.ops.aten.add.Tensor(arg308_1, 1)
        convert_element_type_default: "f32[32, 1280, 7, 7]" = torch.ops.prims.convert_element_type.default(convolution_51, torch.float32)
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_default = None
        getitem_102: "f32[1, 1280, 1, 1]" = var_mean_correction[0]
        getitem_103: "f32[1, 1280, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_154: "f32[1, 1280, 1, 1]" = torch.ops.aten.add.Tensor(getitem_102, 1e-05)
        rsqrt_default: "f32[1, 1280, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor_154);  add_tensor_154 = None
        sub_tensor: "f32[32, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_51, getitem_103);  convolution_51 = None
        mul_tensor_255: "f32[32, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        squeeze_dims_153: "f32[1280]" = torch.ops.aten.squeeze.dims(getitem_103, [0, 2, 3]);  getitem_103 = None
        mul_tensor_256: "f32[1280]" = torch.ops.aten.mul.Tensor(squeeze_dims_153, 0.00029999999999996696);  squeeze_dims_153 = None
        mul_tensor_257: "f32[1280]" = torch.ops.aten.mul.Tensor(arg309_1, 0.9997)
        add_tensor_155: "f32[1280]" = torch.ops.aten.add.Tensor(mul_tensor_256, mul_tensor_257);  mul_tensor_256 = mul_tensor_257 = None
        squeeze_dims_154: "f32[1280]" = torch.ops.aten.squeeze.dims(getitem_102, [0, 2, 3]);  getitem_102 = None
        mul_tensor_258: "f32[1280]" = torch.ops.aten.mul.Tensor(squeeze_dims_154, 1.0006381620931717);  squeeze_dims_154 = None
        mul_tensor_259: "f32[1280]" = torch.ops.aten.mul.Tensor(mul_tensor_258, 0.00029999999999996696);  mul_tensor_258 = None
        mul_tensor_260: "f32[1280]" = torch.ops.aten.mul.Tensor(arg310_1, 0.9997)
        add_tensor_156: "f32[1280]" = torch.ops.aten.add.Tensor(mul_tensor_259, mul_tensor_260);  mul_tensor_259 = mul_tensor_260 = None
        unsqueeze_default: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(arg311_1, -1);  arg311_1 = None
        unsqueeze_default_1: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_261: "f32[32, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_255, unsqueeze_default_1);  mul_tensor_255 = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(arg312_1, -1);  arg312_1 = None
        unsqueeze_default_3: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_157: "f32[32, 1280, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_261, unsqueeze_default_3);  mul_tensor_261 = unsqueeze_default_3 = None
        convert_element_type_default_1: "f16[32, 1280, 7, 7]" = torch.ops.prims.convert_element_type.default(add_tensor_157, torch.float16);  add_tensor_157 = None
        relu_default: "f16[32, 1280, 7, 7]" = torch.ops.aten.relu.default(convert_element_type_default_1);  convert_element_type_default_1 = None
        mean_dim: "f16[32, 1280]" = torch.ops.aten.mean.dim(relu_default, [2, 3]);  relu_default = None
        inductor_seeds_default: "i64[1]" = torch.ops.prims.inductor_seeds.default(1, device(type='cuda', index=0))
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0);  inductor_seeds_default = None
        inductor_random_default: "f32[32, 1280]" = torch.ops.prims.inductor_random.default([32, 1280], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        lt_scalar: "b8[32, 1280]" = torch.ops.aten.lt.Scalar(inductor_random_default, 0.8);  inductor_random_default = None
        convert_element_type_default_2: "f16[32, 1280]" = torch.ops.prims.convert_element_type.default(lt_scalar, torch.float16);  lt_scalar = None
        div_scalar: "f16[32, 1280]" = torch.ops.aten.div.Scalar(convert_element_type_default_2, 0.8);  convert_element_type_default_2 = None
        mul_tensor_262: "f16[32, 1280]" = torch.ops.aten.mul.Tensor(mean_dim, div_scalar);  mean_dim = div_scalar = None
        convert_element_type_default_3: "f16[1000]" = torch.ops.prims.convert_element_type.default(arg314_1, torch.float16);  arg314_1 = None
        convert_element_type_default_4: "f16[1000, 1280]" = torch.ops.prims.convert_element_type.default(arg313_1, torch.float16);  arg313_1 = None
        permute_default: "f16[1280, 1000]" = torch.ops.aten.permute.default(convert_element_type_default_4, [1, 0]);  convert_element_type_default_4 = None
        permute_default_1: "f16[1000, 1280]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        unsqueeze_default_4: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(squeeze_dims_150, 0);  squeeze_dims_150 = None
        unsqueeze_default_5: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 2);  unsqueeze_default_4 = None
        unsqueeze_default_6: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_5, 3);  unsqueeze_default_5 = None
        unsqueeze_default_7: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(squeeze_dims_147, 0);  squeeze_dims_147 = None
        unsqueeze_default_8: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 2);  unsqueeze_default_7 = None
        unsqueeze_default_9: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, 3);  unsqueeze_default_8 = None
        unsqueeze_default_10: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(squeeze_dims_144, 0);  squeeze_dims_144 = None
        unsqueeze_default_11: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 2);  unsqueeze_default_10 = None
        unsqueeze_default_12: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, 3);  unsqueeze_default_11 = None
        unsqueeze_default_13: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_dims_141, 0);  squeeze_dims_141 = None
        unsqueeze_default_14: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 2);  unsqueeze_default_13 = None
        unsqueeze_default_15: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, 3);  unsqueeze_default_14 = None
        unsqueeze_default_16: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(squeeze_dims_138, 0);  squeeze_dims_138 = None
        unsqueeze_default_17: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_16, 2);  unsqueeze_default_16 = None
        unsqueeze_default_18: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_17, 3);  unsqueeze_default_17 = None
        unsqueeze_default_19: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(squeeze_dims_135, 0);  squeeze_dims_135 = None
        unsqueeze_default_20: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_19, 2);  unsqueeze_default_19 = None
        unsqueeze_default_21: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_20, 3);  unsqueeze_default_20 = None
        unsqueeze_default_22: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_dims_132, 0);  squeeze_dims_132 = None
        unsqueeze_default_23: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_22, 2);  unsqueeze_default_22 = None
        unsqueeze_default_24: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_23, 3);  unsqueeze_default_23 = None
        unsqueeze_default_25: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(squeeze_dims_129, 0);  squeeze_dims_129 = None
        unsqueeze_default_26: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_25, 2);  unsqueeze_default_25 = None
        unsqueeze_default_27: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_26, 3);  unsqueeze_default_26 = None
        unsqueeze_default_28: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(squeeze_dims_126, 0);  squeeze_dims_126 = None
        unsqueeze_default_29: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_28, 2);  unsqueeze_default_28 = None
        unsqueeze_default_30: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_29, 3);  unsqueeze_default_29 = None
        unsqueeze_default_31: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_dims_123, 0);  squeeze_dims_123 = None
        unsqueeze_default_32: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_31, 2);  unsqueeze_default_31 = None
        unsqueeze_default_33: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_32, 3);  unsqueeze_default_32 = None
        unsqueeze_default_34: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(squeeze_dims_120, 0);  squeeze_dims_120 = None
        unsqueeze_default_35: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_34, 2);  unsqueeze_default_34 = None
        unsqueeze_default_36: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_35, 3);  unsqueeze_default_35 = None
        unsqueeze_default_37: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(squeeze_dims_117, 0);  squeeze_dims_117 = None
        unsqueeze_default_38: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_37, 2);  unsqueeze_default_37 = None
        unsqueeze_default_39: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_38, 3);  unsqueeze_default_38 = None
        unsqueeze_default_40: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_dims_114, 0);  squeeze_dims_114 = None
        unsqueeze_default_41: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_40, 2);  unsqueeze_default_40 = None
        unsqueeze_default_42: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_41, 3);  unsqueeze_default_41 = None
        unsqueeze_default_43: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(squeeze_dims_111, 0);  squeeze_dims_111 = None
        unsqueeze_default_44: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_43, 2);  unsqueeze_default_43 = None
        unsqueeze_default_45: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_44, 3);  unsqueeze_default_44 = None
        unsqueeze_default_46: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(squeeze_dims_108, 0);  squeeze_dims_108 = None
        unsqueeze_default_47: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_46, 2);  unsqueeze_default_46 = None
        unsqueeze_default_48: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_47, 3);  unsqueeze_default_47 = None
        unsqueeze_default_49: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(squeeze_dims_105, 0);  squeeze_dims_105 = None
        unsqueeze_default_50: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_49, 2);  unsqueeze_default_49 = None
        unsqueeze_default_51: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_50, 3);  unsqueeze_default_50 = None
        unsqueeze_default_52: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(squeeze_dims_102, 0);  squeeze_dims_102 = None
        unsqueeze_default_53: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_52, 2);  unsqueeze_default_52 = None
        unsqueeze_default_54: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_53, 3);  unsqueeze_default_53 = None
        unsqueeze_default_55: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(squeeze_dims_99, 0);  squeeze_dims_99 = None
        unsqueeze_default_56: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_55, 2);  unsqueeze_default_55 = None
        unsqueeze_default_57: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_56, 3);  unsqueeze_default_56 = None
        unsqueeze_default_58: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(squeeze_dims_96, 0);  squeeze_dims_96 = None
        unsqueeze_default_59: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_58, 2);  unsqueeze_default_58 = None
        unsqueeze_default_60: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_59, 3);  unsqueeze_default_59 = None
        unsqueeze_default_61: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(squeeze_dims_93, 0);  squeeze_dims_93 = None
        unsqueeze_default_62: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_61, 2);  unsqueeze_default_61 = None
        unsqueeze_default_63: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_62, 3);  unsqueeze_default_62 = None
        unsqueeze_default_64: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(squeeze_dims_90, 0);  squeeze_dims_90 = None
        unsqueeze_default_65: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_64, 2);  unsqueeze_default_64 = None
        unsqueeze_default_66: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_65, 3);  unsqueeze_default_65 = None
        unsqueeze_default_67: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(squeeze_dims_87, 0);  squeeze_dims_87 = None
        unsqueeze_default_68: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_67, 2);  unsqueeze_default_67 = None
        unsqueeze_default_69: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_68, 3);  unsqueeze_default_68 = None
        unsqueeze_default_70: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(squeeze_dims_84, 0);  squeeze_dims_84 = None
        unsqueeze_default_71: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_70, 2);  unsqueeze_default_70 = None
        unsqueeze_default_72: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_71, 3);  unsqueeze_default_71 = None
        unsqueeze_default_73: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(squeeze_dims_81, 0);  squeeze_dims_81 = None
        unsqueeze_default_74: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_73, 2);  unsqueeze_default_73 = None
        unsqueeze_default_75: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_74, 3);  unsqueeze_default_74 = None
        unsqueeze_default_76: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(squeeze_dims_78, 0);  squeeze_dims_78 = None
        unsqueeze_default_77: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_76, 2);  unsqueeze_default_76 = None
        unsqueeze_default_78: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_77, 3);  unsqueeze_default_77 = None
        unsqueeze_default_79: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(squeeze_dims_75, 0);  squeeze_dims_75 = None
        unsqueeze_default_80: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_79, 2);  unsqueeze_default_79 = None
        unsqueeze_default_81: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_80, 3);  unsqueeze_default_80 = None
        unsqueeze_default_82: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(squeeze_dims_72, 0);  squeeze_dims_72 = None
        unsqueeze_default_83: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_82, 2);  unsqueeze_default_82 = None
        unsqueeze_default_84: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_83, 3);  unsqueeze_default_83 = None
        unsqueeze_default_85: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(squeeze_dims_69, 0);  squeeze_dims_69 = None
        unsqueeze_default_86: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_85, 2);  unsqueeze_default_85 = None
        unsqueeze_default_87: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_86, 3);  unsqueeze_default_86 = None
        unsqueeze_default_88: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(squeeze_dims_66, 0);  squeeze_dims_66 = None
        unsqueeze_default_89: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_88, 2);  unsqueeze_default_88 = None
        unsqueeze_default_90: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_89, 3);  unsqueeze_default_89 = None
        unsqueeze_default_91: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(squeeze_dims_63, 0);  squeeze_dims_63 = None
        unsqueeze_default_92: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_91, 2);  unsqueeze_default_91 = None
        unsqueeze_default_93: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_92, 3);  unsqueeze_default_92 = None
        unsqueeze_default_94: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(squeeze_dims_60, 0);  squeeze_dims_60 = None
        unsqueeze_default_95: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_94, 2);  unsqueeze_default_94 = None
        unsqueeze_default_96: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_95, 3);  unsqueeze_default_95 = None
        unsqueeze_default_97: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(squeeze_dims_57, 0);  squeeze_dims_57 = None
        unsqueeze_default_98: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_97, 2);  unsqueeze_default_97 = None
        unsqueeze_default_99: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_98, 3);  unsqueeze_default_98 = None
        unsqueeze_default_100: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(squeeze_dims_54, 0);  squeeze_dims_54 = None
        unsqueeze_default_101: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_100, 2);  unsqueeze_default_100 = None
        unsqueeze_default_102: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_101, 3);  unsqueeze_default_101 = None
        unsqueeze_default_103: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(squeeze_dims_51, 0);  squeeze_dims_51 = None
        unsqueeze_default_104: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_103, 2);  unsqueeze_default_103 = None
        unsqueeze_default_105: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_104, 3);  unsqueeze_default_104 = None
        unsqueeze_default_106: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(squeeze_dims_48, 0);  squeeze_dims_48 = None
        unsqueeze_default_107: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_106, 2);  unsqueeze_default_106 = None
        unsqueeze_default_108: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_107, 3);  unsqueeze_default_107 = None
        unsqueeze_default_109: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(squeeze_dims_45, 0);  squeeze_dims_45 = None
        unsqueeze_default_110: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_109, 2);  unsqueeze_default_109 = None
        unsqueeze_default_111: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_110, 3);  unsqueeze_default_110 = None
        unsqueeze_default_112: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(squeeze_dims_42, 0);  squeeze_dims_42 = None
        unsqueeze_default_113: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_112, 2);  unsqueeze_default_112 = None
        unsqueeze_default_114: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_113, 3);  unsqueeze_default_113 = None
        unsqueeze_default_115: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(squeeze_dims_39, 0);  squeeze_dims_39 = None
        unsqueeze_default_116: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_115, 2);  unsqueeze_default_115 = None
        unsqueeze_default_117: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_116, 3);  unsqueeze_default_116 = None
        unsqueeze_default_118: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(squeeze_dims_36, 0);  squeeze_dims_36 = None
        unsqueeze_default_119: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_118, 2);  unsqueeze_default_118 = None
        unsqueeze_default_120: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_119, 3);  unsqueeze_default_119 = None
        unsqueeze_default_121: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(squeeze_dims_33, 0);  squeeze_dims_33 = None
        unsqueeze_default_122: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_121, 2);  unsqueeze_default_121 = None
        unsqueeze_default_123: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_122, 3);  unsqueeze_default_122 = None
        unsqueeze_default_124: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(squeeze_dims_30, 0);  squeeze_dims_30 = None
        unsqueeze_default_125: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_124, 2);  unsqueeze_default_124 = None
        unsqueeze_default_126: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_125, 3);  unsqueeze_default_125 = None
        unsqueeze_default_127: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(squeeze_dims_27, 0);  squeeze_dims_27 = None
        unsqueeze_default_128: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_127, 2);  unsqueeze_default_127 = None
        unsqueeze_default_129: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_128, 3);  unsqueeze_default_128 = None
        unsqueeze_default_130: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(squeeze_dims_24, 0);  squeeze_dims_24 = None
        unsqueeze_default_131: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_130, 2);  unsqueeze_default_130 = None
        unsqueeze_default_132: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_131, 3);  unsqueeze_default_131 = None
        unsqueeze_default_133: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(squeeze_dims_21, 0);  squeeze_dims_21 = None
        unsqueeze_default_134: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_133, 2);  unsqueeze_default_133 = None
        unsqueeze_default_135: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_134, 3);  unsqueeze_default_134 = None
        unsqueeze_default_136: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(squeeze_dims_18, 0);  squeeze_dims_18 = None
        unsqueeze_default_137: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_136, 2);  unsqueeze_default_136 = None
        unsqueeze_default_138: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_137, 3);  unsqueeze_default_137 = None
        unsqueeze_default_139: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(squeeze_dims_15, 0);  squeeze_dims_15 = None
        unsqueeze_default_140: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_139, 2);  unsqueeze_default_139 = None
        unsqueeze_default_141: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_140, 3);  unsqueeze_default_140 = None
        unsqueeze_default_142: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(squeeze_dims_12, 0);  squeeze_dims_12 = None
        unsqueeze_default_143: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_142, 2);  unsqueeze_default_142 = None
        unsqueeze_default_144: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_143, 3);  unsqueeze_default_143 = None
        unsqueeze_default_145: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(squeeze_dims_9, 0);  squeeze_dims_9 = None
        unsqueeze_default_146: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_145, 2);  unsqueeze_default_145 = None
        unsqueeze_default_147: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_146, 3);  unsqueeze_default_146 = None
        unsqueeze_default_148: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(squeeze_dims_6, 0);  squeeze_dims_6 = None
        unsqueeze_default_149: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_148, 2);  unsqueeze_default_148 = None
        unsqueeze_default_150: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_149, 3);  unsqueeze_default_149 = None
        unsqueeze_default_151: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_dims_3, 0);  squeeze_dims_3 = None
        unsqueeze_default_152: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_151, 2);  unsqueeze_default_151 = None
        unsqueeze_default_153: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_152, 3);  unsqueeze_default_152 = None
        unsqueeze_default_154: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_155: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_154, 2);  unsqueeze_default_154 = None
        unsqueeze_default_156: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_155, 3);  unsqueeze_default_155 = None
        copy__default: "i64[]" = torch.ops.aten.copy_.default(arg2_1, add_tensor);  arg2_1 = add_tensor = None
        copy__default_1: "f32[32]" = torch.ops.aten.copy_.default(arg3_1, add_tensor_1);  arg3_1 = add_tensor_1 = None
        copy__default_2: "f32[32]" = torch.ops.aten.copy_.default(arg4_1, add_tensor_2);  arg4_1 = add_tensor_2 = None
        copy__default_3: "i64[]" = torch.ops.aten.copy_.default(arg8_1, add_tensor_3);  arg8_1 = add_tensor_3 = None
        copy__default_4: "f32[32]" = torch.ops.aten.copy_.default(arg9_1, add_tensor_4);  arg9_1 = add_tensor_4 = None
        copy__default_5: "f32[32]" = torch.ops.aten.copy_.default(arg10_1, add_tensor_5);  arg10_1 = add_tensor_5 = None
        copy__default_6: "i64[]" = torch.ops.aten.copy_.default(arg14_1, add_tensor_6);  arg14_1 = add_tensor_6 = None
        copy__default_7: "f32[16]" = torch.ops.aten.copy_.default(arg15_1, add_tensor_7);  arg15_1 = add_tensor_7 = None
        copy__default_8: "f32[16]" = torch.ops.aten.copy_.default(arg16_1, add_tensor_8);  arg16_1 = add_tensor_8 = None
        copy__default_9: "i64[]" = torch.ops.aten.copy_.default(arg20_1, add_tensor_9);  arg20_1 = add_tensor_9 = None
        copy__default_10: "f32[48]" = torch.ops.aten.copy_.default(arg21_1, add_tensor_10);  arg21_1 = add_tensor_10 = None
        copy__default_11: "f32[48]" = torch.ops.aten.copy_.default(arg22_1, add_tensor_11);  arg22_1 = add_tensor_11 = None
        copy__default_12: "i64[]" = torch.ops.aten.copy_.default(arg26_1, add_tensor_12);  arg26_1 = add_tensor_12 = None
        copy__default_13: "f32[48]" = torch.ops.aten.copy_.default(arg27_1, add_tensor_13);  arg27_1 = add_tensor_13 = None
        copy__default_14: "f32[48]" = torch.ops.aten.copy_.default(arg28_1, add_tensor_14);  arg28_1 = add_tensor_14 = None
        copy__default_15: "i64[]" = torch.ops.aten.copy_.default(arg32_1, add_tensor_15);  arg32_1 = add_tensor_15 = None
        copy__default_16: "f32[24]" = torch.ops.aten.copy_.default(arg33_1, add_tensor_16);  arg33_1 = add_tensor_16 = None
        copy__default_17: "f32[24]" = torch.ops.aten.copy_.default(arg34_1, add_tensor_17);  arg34_1 = add_tensor_17 = None
        copy__default_18: "i64[]" = torch.ops.aten.copy_.default(arg38_1, add_tensor_18);  arg38_1 = add_tensor_18 = None
        copy__default_19: "f32[72]" = torch.ops.aten.copy_.default(arg39_1, add_tensor_19);  arg39_1 = add_tensor_19 = None
        copy__default_20: "f32[72]" = torch.ops.aten.copy_.default(arg40_1, add_tensor_20);  arg40_1 = add_tensor_20 = None
        copy__default_21: "i64[]" = torch.ops.aten.copy_.default(arg44_1, add_tensor_21);  arg44_1 = add_tensor_21 = None
        copy__default_22: "f32[72]" = torch.ops.aten.copy_.default(arg45_1, add_tensor_22);  arg45_1 = add_tensor_22 = None
        copy__default_23: "f32[72]" = torch.ops.aten.copy_.default(arg46_1, add_tensor_23);  arg46_1 = add_tensor_23 = None
        copy__default_24: "i64[]" = torch.ops.aten.copy_.default(arg50_1, add_tensor_24);  arg50_1 = add_tensor_24 = None
        copy__default_25: "f32[24]" = torch.ops.aten.copy_.default(arg51_1, add_tensor_25);  arg51_1 = add_tensor_25 = None
        copy__default_26: "f32[24]" = torch.ops.aten.copy_.default(arg52_1, add_tensor_26);  arg52_1 = add_tensor_26 = None
        copy__default_27: "i64[]" = torch.ops.aten.copy_.default(arg56_1, add_tensor_27);  arg56_1 = add_tensor_27 = None
        copy__default_28: "f32[72]" = torch.ops.aten.copy_.default(arg57_1, add_tensor_28);  arg57_1 = add_tensor_28 = None
        copy__default_29: "f32[72]" = torch.ops.aten.copy_.default(arg58_1, add_tensor_29);  arg58_1 = add_tensor_29 = None
        copy__default_30: "i64[]" = torch.ops.aten.copy_.default(arg62_1, add_tensor_30);  arg62_1 = add_tensor_30 = None
        copy__default_31: "f32[72]" = torch.ops.aten.copy_.default(arg63_1, add_tensor_31);  arg63_1 = add_tensor_31 = None
        copy__default_32: "f32[72]" = torch.ops.aten.copy_.default(arg64_1, add_tensor_32);  arg64_1 = add_tensor_32 = None
        copy__default_33: "i64[]" = torch.ops.aten.copy_.default(arg68_1, add_tensor_33);  arg68_1 = add_tensor_33 = None
        copy__default_34: "f32[24]" = torch.ops.aten.copy_.default(arg69_1, add_tensor_34);  arg69_1 = add_tensor_34 = None
        copy__default_35: "f32[24]" = torch.ops.aten.copy_.default(arg70_1, add_tensor_35);  arg70_1 = add_tensor_35 = None
        copy__default_36: "i64[]" = torch.ops.aten.copy_.default(arg74_1, add_tensor_36);  arg74_1 = add_tensor_36 = None
        copy__default_37: "f32[72]" = torch.ops.aten.copy_.default(arg75_1, add_tensor_37);  arg75_1 = add_tensor_37 = None
        copy__default_38: "f32[72]" = torch.ops.aten.copy_.default(arg76_1, add_tensor_38);  arg76_1 = add_tensor_38 = None
        copy__default_39: "i64[]" = torch.ops.aten.copy_.default(arg80_1, add_tensor_39);  arg80_1 = add_tensor_39 = None
        copy__default_40: "f32[72]" = torch.ops.aten.copy_.default(arg81_1, add_tensor_40);  arg81_1 = add_tensor_40 = None
        copy__default_41: "f32[72]" = torch.ops.aten.copy_.default(arg82_1, add_tensor_41);  arg82_1 = add_tensor_41 = None
        copy__default_42: "i64[]" = torch.ops.aten.copy_.default(arg86_1, add_tensor_42);  arg86_1 = add_tensor_42 = None
        copy__default_43: "f32[40]" = torch.ops.aten.copy_.default(arg87_1, add_tensor_43);  arg87_1 = add_tensor_43 = None
        copy__default_44: "f32[40]" = torch.ops.aten.copy_.default(arg88_1, add_tensor_44);  arg88_1 = add_tensor_44 = None
        copy__default_45: "i64[]" = torch.ops.aten.copy_.default(arg92_1, add_tensor_45);  arg92_1 = add_tensor_45 = None
        copy__default_46: "f32[120]" = torch.ops.aten.copy_.default(arg93_1, add_tensor_46);  arg93_1 = add_tensor_46 = None
        copy__default_47: "f32[120]" = torch.ops.aten.copy_.default(arg94_1, add_tensor_47);  arg94_1 = add_tensor_47 = None
        copy__default_48: "i64[]" = torch.ops.aten.copy_.default(arg98_1, add_tensor_48);  arg98_1 = add_tensor_48 = None
        copy__default_49: "f32[120]" = torch.ops.aten.copy_.default(arg99_1, add_tensor_49);  arg99_1 = add_tensor_49 = None
        copy__default_50: "f32[120]" = torch.ops.aten.copy_.default(arg100_1, add_tensor_50);  arg100_1 = add_tensor_50 = None
        copy__default_51: "i64[]" = torch.ops.aten.copy_.default(arg104_1, add_tensor_51);  arg104_1 = add_tensor_51 = None
        copy__default_52: "f32[40]" = torch.ops.aten.copy_.default(arg105_1, add_tensor_52);  arg105_1 = add_tensor_52 = None
        copy__default_53: "f32[40]" = torch.ops.aten.copy_.default(arg106_1, add_tensor_53);  arg106_1 = add_tensor_53 = None
        copy__default_54: "i64[]" = torch.ops.aten.copy_.default(arg110_1, add_tensor_54);  arg110_1 = add_tensor_54 = None
        copy__default_55: "f32[120]" = torch.ops.aten.copy_.default(arg111_1, add_tensor_55);  arg111_1 = add_tensor_55 = None
        copy__default_56: "f32[120]" = torch.ops.aten.copy_.default(arg112_1, add_tensor_56);  arg112_1 = add_tensor_56 = None
        copy__default_57: "i64[]" = torch.ops.aten.copy_.default(arg116_1, add_tensor_57);  arg116_1 = add_tensor_57 = None
        copy__default_58: "f32[120]" = torch.ops.aten.copy_.default(arg117_1, add_tensor_58);  arg117_1 = add_tensor_58 = None
        copy__default_59: "f32[120]" = torch.ops.aten.copy_.default(arg118_1, add_tensor_59);  arg118_1 = add_tensor_59 = None
        copy__default_60: "i64[]" = torch.ops.aten.copy_.default(arg122_1, add_tensor_60);  arg122_1 = add_tensor_60 = None
        copy__default_61: "f32[40]" = torch.ops.aten.copy_.default(arg123_1, add_tensor_61);  arg123_1 = add_tensor_61 = None
        copy__default_62: "f32[40]" = torch.ops.aten.copy_.default(arg124_1, add_tensor_62);  arg124_1 = add_tensor_62 = None
        copy__default_63: "i64[]" = torch.ops.aten.copy_.default(arg128_1, add_tensor_63);  arg128_1 = add_tensor_63 = None
        copy__default_64: "f32[240]" = torch.ops.aten.copy_.default(arg129_1, add_tensor_64);  arg129_1 = add_tensor_64 = None
        copy__default_65: "f32[240]" = torch.ops.aten.copy_.default(arg130_1, add_tensor_65);  arg130_1 = add_tensor_65 = None
        copy__default_66: "i64[]" = torch.ops.aten.copy_.default(arg134_1, add_tensor_66);  arg134_1 = add_tensor_66 = None
        copy__default_67: "f32[240]" = torch.ops.aten.copy_.default(arg135_1, add_tensor_67);  arg135_1 = add_tensor_67 = None
        copy__default_68: "f32[240]" = torch.ops.aten.copy_.default(arg136_1, add_tensor_68);  arg136_1 = add_tensor_68 = None
        copy__default_69: "i64[]" = torch.ops.aten.copy_.default(arg140_1, add_tensor_69);  arg140_1 = add_tensor_69 = None
        copy__default_70: "f32[80]" = torch.ops.aten.copy_.default(arg141_1, add_tensor_70);  arg141_1 = add_tensor_70 = None
        copy__default_71: "f32[80]" = torch.ops.aten.copy_.default(arg142_1, add_tensor_71);  arg142_1 = add_tensor_71 = None
        copy__default_72: "i64[]" = torch.ops.aten.copy_.default(arg146_1, add_tensor_72);  arg146_1 = add_tensor_72 = None
        copy__default_73: "f32[480]" = torch.ops.aten.copy_.default(arg147_1, add_tensor_73);  arg147_1 = add_tensor_73 = None
        copy__default_74: "f32[480]" = torch.ops.aten.copy_.default(arg148_1, add_tensor_74);  arg148_1 = add_tensor_74 = None
        copy__default_75: "i64[]" = torch.ops.aten.copy_.default(arg152_1, add_tensor_75);  arg152_1 = add_tensor_75 = None
        copy__default_76: "f32[480]" = torch.ops.aten.copy_.default(arg153_1, add_tensor_76);  arg153_1 = add_tensor_76 = None
        copy__default_77: "f32[480]" = torch.ops.aten.copy_.default(arg154_1, add_tensor_77);  arg154_1 = add_tensor_77 = None
        copy__default_78: "i64[]" = torch.ops.aten.copy_.default(arg158_1, add_tensor_78);  arg158_1 = add_tensor_78 = None
        copy__default_79: "f32[80]" = torch.ops.aten.copy_.default(arg159_1, add_tensor_79);  arg159_1 = add_tensor_79 = None
        copy__default_80: "f32[80]" = torch.ops.aten.copy_.default(arg160_1, add_tensor_80);  arg160_1 = add_tensor_80 = None
        copy__default_81: "i64[]" = torch.ops.aten.copy_.default(arg164_1, add_tensor_81);  arg164_1 = add_tensor_81 = None
        copy__default_82: "f32[480]" = torch.ops.aten.copy_.default(arg165_1, add_tensor_82);  arg165_1 = add_tensor_82 = None
        copy__default_83: "f32[480]" = torch.ops.aten.copy_.default(arg166_1, add_tensor_83);  arg166_1 = add_tensor_83 = None
        copy__default_84: "i64[]" = torch.ops.aten.copy_.default(arg170_1, add_tensor_84);  arg170_1 = add_tensor_84 = None
        copy__default_85: "f32[480]" = torch.ops.aten.copy_.default(arg171_1, add_tensor_85);  arg171_1 = add_tensor_85 = None
        copy__default_86: "f32[480]" = torch.ops.aten.copy_.default(arg172_1, add_tensor_86);  arg172_1 = add_tensor_86 = None
        copy__default_87: "i64[]" = torch.ops.aten.copy_.default(arg176_1, add_tensor_87);  arg176_1 = add_tensor_87 = None
        copy__default_88: "f32[80]" = torch.ops.aten.copy_.default(arg177_1, add_tensor_88);  arg177_1 = add_tensor_88 = None
        copy__default_89: "f32[80]" = torch.ops.aten.copy_.default(arg178_1, add_tensor_89);  arg178_1 = add_tensor_89 = None
        copy__default_90: "i64[]" = torch.ops.aten.copy_.default(arg182_1, add_tensor_90);  arg182_1 = add_tensor_90 = None
        copy__default_91: "f32[480]" = torch.ops.aten.copy_.default(arg183_1, add_tensor_91);  arg183_1 = add_tensor_91 = None
        copy__default_92: "f32[480]" = torch.ops.aten.copy_.default(arg184_1, add_tensor_92);  arg184_1 = add_tensor_92 = None
        copy__default_93: "i64[]" = torch.ops.aten.copy_.default(arg188_1, add_tensor_93);  arg188_1 = add_tensor_93 = None
        copy__default_94: "f32[480]" = torch.ops.aten.copy_.default(arg189_1, add_tensor_94);  arg189_1 = add_tensor_94 = None
        copy__default_95: "f32[480]" = torch.ops.aten.copy_.default(arg190_1, add_tensor_95);  arg190_1 = add_tensor_95 = None
        copy__default_96: "i64[]" = torch.ops.aten.copy_.default(arg194_1, add_tensor_96);  arg194_1 = add_tensor_96 = None
        copy__default_97: "f32[96]" = torch.ops.aten.copy_.default(arg195_1, add_tensor_97);  arg195_1 = add_tensor_97 = None
        copy__default_98: "f32[96]" = torch.ops.aten.copy_.default(arg196_1, add_tensor_98);  arg196_1 = add_tensor_98 = None
        copy__default_99: "i64[]" = torch.ops.aten.copy_.default(arg200_1, add_tensor_99);  arg200_1 = add_tensor_99 = None
        copy__default_100: "f32[576]" = torch.ops.aten.copy_.default(arg201_1, add_tensor_100);  arg201_1 = add_tensor_100 = None
        copy__default_101: "f32[576]" = torch.ops.aten.copy_.default(arg202_1, add_tensor_101);  arg202_1 = add_tensor_101 = None
        copy__default_102: "i64[]" = torch.ops.aten.copy_.default(arg206_1, add_tensor_102);  arg206_1 = add_tensor_102 = None
        copy__default_103: "f32[576]" = torch.ops.aten.copy_.default(arg207_1, add_tensor_103);  arg207_1 = add_tensor_103 = None
        copy__default_104: "f32[576]" = torch.ops.aten.copy_.default(arg208_1, add_tensor_104);  arg208_1 = add_tensor_104 = None
        copy__default_105: "i64[]" = torch.ops.aten.copy_.default(arg212_1, add_tensor_105);  arg212_1 = add_tensor_105 = None
        copy__default_106: "f32[96]" = torch.ops.aten.copy_.default(arg213_1, add_tensor_106);  arg213_1 = add_tensor_106 = None
        copy__default_107: "f32[96]" = torch.ops.aten.copy_.default(arg214_1, add_tensor_107);  arg214_1 = add_tensor_107 = None
        copy__default_108: "i64[]" = torch.ops.aten.copy_.default(arg218_1, add_tensor_108);  arg218_1 = add_tensor_108 = None
        copy__default_109: "f32[576]" = torch.ops.aten.copy_.default(arg219_1, add_tensor_109);  arg219_1 = add_tensor_109 = None
        copy__default_110: "f32[576]" = torch.ops.aten.copy_.default(arg220_1, add_tensor_110);  arg220_1 = add_tensor_110 = None
        copy__default_111: "i64[]" = torch.ops.aten.copy_.default(arg224_1, add_tensor_111);  arg224_1 = add_tensor_111 = None
        copy__default_112: "f32[576]" = torch.ops.aten.copy_.default(arg225_1, add_tensor_112);  arg225_1 = add_tensor_112 = None
        copy__default_113: "f32[576]" = torch.ops.aten.copy_.default(arg226_1, add_tensor_113);  arg226_1 = add_tensor_113 = None
        copy__default_114: "i64[]" = torch.ops.aten.copy_.default(arg230_1, add_tensor_114);  arg230_1 = add_tensor_114 = None
        copy__default_115: "f32[192]" = torch.ops.aten.copy_.default(arg231_1, add_tensor_115);  arg231_1 = add_tensor_115 = None
        copy__default_116: "f32[192]" = torch.ops.aten.copy_.default(arg232_1, add_tensor_116);  arg232_1 = add_tensor_116 = None
        copy__default_117: "i64[]" = torch.ops.aten.copy_.default(arg236_1, add_tensor_117);  arg236_1 = add_tensor_117 = None
        copy__default_118: "f32[1152]" = torch.ops.aten.copy_.default(arg237_1, add_tensor_118);  arg237_1 = add_tensor_118 = None
        copy__default_119: "f32[1152]" = torch.ops.aten.copy_.default(arg238_1, add_tensor_119);  arg238_1 = add_tensor_119 = None
        copy__default_120: "i64[]" = torch.ops.aten.copy_.default(arg242_1, add_tensor_120);  arg242_1 = add_tensor_120 = None
        copy__default_121: "f32[1152]" = torch.ops.aten.copy_.default(arg243_1, add_tensor_121);  arg243_1 = add_tensor_121 = None
        copy__default_122: "f32[1152]" = torch.ops.aten.copy_.default(arg244_1, add_tensor_122);  arg244_1 = add_tensor_122 = None
        copy__default_123: "i64[]" = torch.ops.aten.copy_.default(arg248_1, add_tensor_123);  arg248_1 = add_tensor_123 = None
        copy__default_124: "f32[192]" = torch.ops.aten.copy_.default(arg249_1, add_tensor_124);  arg249_1 = add_tensor_124 = None
        copy__default_125: "f32[192]" = torch.ops.aten.copy_.default(arg250_1, add_tensor_125);  arg250_1 = add_tensor_125 = None
        copy__default_126: "i64[]" = torch.ops.aten.copy_.default(arg254_1, add_tensor_126);  arg254_1 = add_tensor_126 = None
        copy__default_127: "f32[1152]" = torch.ops.aten.copy_.default(arg255_1, add_tensor_127);  arg255_1 = add_tensor_127 = None
        copy__default_128: "f32[1152]" = torch.ops.aten.copy_.default(arg256_1, add_tensor_128);  arg256_1 = add_tensor_128 = None
        copy__default_129: "i64[]" = torch.ops.aten.copy_.default(arg260_1, add_tensor_129);  arg260_1 = add_tensor_129 = None
        copy__default_130: "f32[1152]" = torch.ops.aten.copy_.default(arg261_1, add_tensor_130);  arg261_1 = add_tensor_130 = None
        copy__default_131: "f32[1152]" = torch.ops.aten.copy_.default(arg262_1, add_tensor_131);  arg262_1 = add_tensor_131 = None
        copy__default_132: "i64[]" = torch.ops.aten.copy_.default(arg266_1, add_tensor_132);  arg266_1 = add_tensor_132 = None
        copy__default_133: "f32[192]" = torch.ops.aten.copy_.default(arg267_1, add_tensor_133);  arg267_1 = add_tensor_133 = None
        copy__default_134: "f32[192]" = torch.ops.aten.copy_.default(arg268_1, add_tensor_134);  arg268_1 = add_tensor_134 = None
        copy__default_135: "i64[]" = torch.ops.aten.copy_.default(arg272_1, add_tensor_135);  arg272_1 = add_tensor_135 = None
        copy__default_136: "f32[1152]" = torch.ops.aten.copy_.default(arg273_1, add_tensor_136);  arg273_1 = add_tensor_136 = None
        copy__default_137: "f32[1152]" = torch.ops.aten.copy_.default(arg274_1, add_tensor_137);  arg274_1 = add_tensor_137 = None
        copy__default_138: "i64[]" = torch.ops.aten.copy_.default(arg278_1, add_tensor_138);  arg278_1 = add_tensor_138 = None
        copy__default_139: "f32[1152]" = torch.ops.aten.copy_.default(arg279_1, add_tensor_139);  arg279_1 = add_tensor_139 = None
        copy__default_140: "f32[1152]" = torch.ops.aten.copy_.default(arg280_1, add_tensor_140);  arg280_1 = add_tensor_140 = None
        copy__default_141: "i64[]" = torch.ops.aten.copy_.default(arg284_1, add_tensor_141);  arg284_1 = add_tensor_141 = None
        copy__default_142: "f32[192]" = torch.ops.aten.copy_.default(arg285_1, add_tensor_142);  arg285_1 = add_tensor_142 = None
        copy__default_143: "f32[192]" = torch.ops.aten.copy_.default(arg286_1, add_tensor_143);  arg286_1 = add_tensor_143 = None
        copy__default_144: "i64[]" = torch.ops.aten.copy_.default(arg290_1, add_tensor_144);  arg290_1 = add_tensor_144 = None
        copy__default_145: "f32[1152]" = torch.ops.aten.copy_.default(arg291_1, add_tensor_145);  arg291_1 = add_tensor_145 = None
        copy__default_146: "f32[1152]" = torch.ops.aten.copy_.default(arg292_1, add_tensor_146);  arg292_1 = add_tensor_146 = None
        copy__default_147: "i64[]" = torch.ops.aten.copy_.default(arg296_1, add_tensor_147);  arg296_1 = add_tensor_147 = None
        copy__default_148: "f32[1152]" = torch.ops.aten.copy_.default(arg297_1, add_tensor_148);  arg297_1 = add_tensor_148 = None
        copy__default_149: "f32[1152]" = torch.ops.aten.copy_.default(arg298_1, add_tensor_149);  arg298_1 = add_tensor_149 = None
        copy__default_150: "i64[]" = torch.ops.aten.copy_.default(arg302_1, add_tensor_150);  arg302_1 = add_tensor_150 = None
        copy__default_151: "f32[320]" = torch.ops.aten.copy_.default(arg303_1, add_tensor_151);  arg303_1 = add_tensor_151 = None
        copy__default_152: "f32[320]" = torch.ops.aten.copy_.default(arg304_1, add_tensor_152);  arg304_1 = add_tensor_152 = None
        copy__default_153: "i64[]" = torch.ops.aten.copy_.default(arg308_1, add_tensor_153);  arg308_1 = add_tensor_153 = None
        copy__default_154: "f32[1280]" = torch.ops.aten.copy_.default(arg309_1, add_tensor_155);  arg309_1 = add_tensor_155 = None
        copy__default_155: "f32[1280]" = torch.ops.aten.copy_.default(arg310_1, add_tensor_156);  arg310_1 = add_tensor_156 = None
        return (squeeze_dims_1, squeeze_dims_4, squeeze_dims_7, squeeze_dims_10, squeeze_dims_13, squeeze_dims_16, squeeze_dims_19, squeeze_dims_22, squeeze_dims_25, squeeze_dims_28, squeeze_dims_31, squeeze_dims_34, squeeze_dims_37, squeeze_dims_40, squeeze_dims_43, squeeze_dims_46, squeeze_dims_49, squeeze_dims_52, squeeze_dims_55, squeeze_dims_58, squeeze_dims_61, squeeze_dims_64, squeeze_dims_67, squeeze_dims_70, squeeze_dims_73, squeeze_dims_76, squeeze_dims_79, squeeze_dims_82, squeeze_dims_85, squeeze_dims_88, squeeze_dims_91, squeeze_dims_94, squeeze_dims_97, squeeze_dims_100, squeeze_dims_103, squeeze_dims_106, squeeze_dims_109, squeeze_dims_112, squeeze_dims_115, squeeze_dims_118, squeeze_dims_121, squeeze_dims_124, squeeze_dims_127, squeeze_dims_130, squeeze_dims_133, squeeze_dims_136, squeeze_dims_139, squeeze_dims_142, squeeze_dims_145, squeeze_dims_148, squeeze_dims_151, mul_tensor_262, convert_element_type_default_3, permute_default_1, unsqueeze_default_6, unsqueeze_default_9, unsqueeze_default_12, unsqueeze_default_15, unsqueeze_default_18, unsqueeze_default_21, unsqueeze_default_24, unsqueeze_default_27, unsqueeze_default_30, unsqueeze_default_33, unsqueeze_default_36, unsqueeze_default_39, unsqueeze_default_42, unsqueeze_default_45, unsqueeze_default_48, unsqueeze_default_51, unsqueeze_default_54, unsqueeze_default_57, unsqueeze_default_60, unsqueeze_default_63, unsqueeze_default_66, unsqueeze_default_69, unsqueeze_default_72, unsqueeze_default_75, unsqueeze_default_78, unsqueeze_default_81, unsqueeze_default_84, unsqueeze_default_87, unsqueeze_default_90, unsqueeze_default_93, unsqueeze_default_96, unsqueeze_default_99, unsqueeze_default_102, unsqueeze_default_105, unsqueeze_default_108, unsqueeze_default_111, unsqueeze_default_114, unsqueeze_default_117, unsqueeze_default_120, unsqueeze_default_123, unsqueeze_default_126, unsqueeze_default_129, unsqueeze_default_132, unsqueeze_default_135, unsqueeze_default_138, unsqueeze_default_141, unsqueeze_default_144, unsqueeze_default_147, unsqueeze_default_150, unsqueeze_default_153, unsqueeze_default_156, copy__default, copy__default_1, copy__default_2, copy__default_3, copy__default_4, copy__default_5, copy__default_6, copy__default_7, copy__default_8, copy__default_9, copy__default_10, copy__default_11, copy__default_12, copy__default_13, copy__default_14, copy__default_15, copy__default_16, copy__default_17, copy__default_18, copy__default_19, copy__default_20, copy__default_21, copy__default_22, copy__default_23, copy__default_24, copy__default_25, copy__default_26, copy__default_27, copy__default_28, copy__default_29, copy__default_30, copy__default_31, copy__default_32, copy__default_33, copy__default_34, copy__default_35, copy__default_36, copy__default_37, copy__default_38, copy__default_39, copy__default_40, copy__default_41, copy__default_42, copy__default_43, copy__default_44, copy__default_45, copy__default_46, copy__default_47, copy__default_48, copy__default_49, copy__default_50, copy__default_51, copy__default_52, copy__default_53, copy__default_54, copy__default_55, copy__default_56, copy__default_57, copy__default_58, copy__default_59, copy__default_60, copy__default_61, copy__default_62, copy__default_63, copy__default_64, copy__default_65, copy__default_66, copy__default_67, copy__default_68, copy__default_69, copy__default_70, copy__default_71, copy__default_72, copy__default_73, copy__default_74, copy__default_75, copy__default_76, copy__default_77, copy__default_78, copy__default_79, copy__default_80, copy__default_81, copy__default_82, copy__default_83, copy__default_84, copy__default_85, copy__default_86, copy__default_87, copy__default_88, copy__default_89, copy__default_90, copy__default_91, copy__default_92, copy__default_93, copy__default_94, copy__default_95, copy__default_96, copy__default_97, copy__default_98, copy__default_99, copy__default_100, copy__default_101, copy__default_102, copy__default_103, copy__default_104, copy__default_105, copy__default_106, copy__default_107, copy__default_108, copy__default_109, copy__default_110, copy__default_111, copy__default_112, copy__default_113, copy__default_114, copy__default_115, copy__default_116, copy__default_117, copy__default_118, copy__default_119, copy__default_120, copy__default_121, copy__default_122, copy__default_123, copy__default_124, copy__default_125, copy__default_126, copy__default_127, copy__default_128, copy__default_129, copy__default_130, copy__default_131, copy__default_132, copy__default_133, copy__default_134, copy__default_135, copy__default_136, copy__default_137, copy__default_138, copy__default_139, copy__default_140, copy__default_141, copy__default_142, copy__default_143, copy__default_144, copy__default_145, copy__default_146, copy__default_147, copy__default_148, copy__default_149, copy__default_150, copy__default_151, copy__default_152, copy__default_153, copy__default_154, copy__default_155)


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
