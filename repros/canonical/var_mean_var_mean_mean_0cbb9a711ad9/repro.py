"""
Standalone repro captured via capture_hook.
Label: inductor_timm_perf-5-6-linux.aws.a100_graph18
Pattern hash: 0cbb9a711ad9
Shape hash: 81f666dd
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([], i64, max=2), T([1, 64, 1, 1], f32), T([1, 64, 1, 1], f32), T([64], f32), T([1, 64, 1, 1], f32), T([64], f32), T([], i64, max=2), T([1, 64, 1, 1], f32), T([1, 64, 1, 1], f32), T([64], f32), T([1, 64, 1, 1], f32), T([64], f32), T([], i64, max=2), T([1, 96, 1, 1], f32), T([1, 96, 1, 1], f32), T([96], f32), T([1, 96, 1, 1], f32), T([96], f32), T([], i64, max=2), T([1, 96, 1, 1], f32), T([1, 96, 1, 1], f32), T([96], f32), T([1, 96, 1, 1], f32), T([96], f32), T([], i64, max=2), T([1, 96, 1, 1], f32), T([1, 96, 1, 1], f32), T([96], f32), T([1, 96, 1, 1], f32), T([96], f32), T([], i64, max=2), T([1, 96, 1, 1], f32), T([1, 96, 1, 1], f32), T([96], f32), T([1, 96, 1, 1], f32), T([96], f32), T([], i64, max=2), T([1, 96, 1, 1], f32), T([1, 96, 1, 1], f32), T([96], f32), T([1, 96, 1, 1], f32), T([96], f32), T([], i64, max=2), T([1, 192, 1, 1], f32), T([1, 192, 1, 1], f32), T([192], f32), T([1, 192, 1, 1], f32), T([192], f32), T([], i64, max=2), T([1, 192, 1, 1], f32), T([1, 192, 1, 1], f32), T([192], f32), T([1, 192, 1, 1], f32), T([192], f32), T([], i64, max=2), T([1, 192, 1, 1], f32), T([1, 192, 1, 1], f32), T([192], f32), T([1, 192, 1, 1], f32), T([192], f32), T([], i64, max=2), T([1, 192, 1, 1], f32), T([1, 192, 1, 1], f32), T([192], f32), T([1, 192, 1, 1], f32), T([192], f32), T([], i64, max=2), T([1, 192, 1, 1], f32), T([1, 192, 1, 1], f32), T([192], f32), T([1, 192, 1, 1], f32), T([192], f32), T([], i64, max=2), T([1, 192, 1, 1], f32), T([1, 192, 1, 1], f32), T([192], f32), T([1, 192, 1, 1], f32), T([192], f32), T([], i64, max=2), T([1, 192, 1, 1], f32), T([1, 192, 1, 1], f32), T([192], f32), T([1, 192, 1, 1], f32), T([192], f32), T([], i64, max=2), T([1, 192, 1, 1], f32), T([1, 192, 1, 1], f32), T([192], f32), T([1, 192, 1, 1], f32), T([192], f32), T([], i64, max=2), T([1, 192, 1, 1], f32), T([1, 192, 1, 1], f32), T([192], f32), T([1, 192, 1, 1], f32), T([192], f32), T([], i64, max=2), T([1, 192, 1, 1], f32), T([1, 192, 1, 1], f32), T([192], f32), T([1, 192, 1, 1], f32), T([192], f32), T([], i64, max=2), T([1, 192, 1, 1], f32), T([1, 192, 1, 1], f32), T([192], f32), T([1, 192, 1, 1], f32), T([192], f32), T([], i64, max=2), T([1, 384, 1, 1], f32), T([1, 384, 1, 1], f32), T([384], f32), T([1, 384, 1, 1], f32), T([384], f32), T([], i64, max=2), T([1, 384, 1, 1], f32), T([1, 384, 1, 1], f32), T([384], f32), T([1, 384, 1, 1], f32), T([384], f32), T([], i64, max=2), T([1, 384, 1, 1], f32), T([1, 384, 1, 1], f32), T([384], f32), T([1, 384, 1, 1], f32), T([384], f32), T([], i64, max=2), T([1, 384, 1, 1], f32), T([1, 384, 1, 1], f32), T([384], f32), T([1, 384, 1, 1], f32), T([384], f32), T([], i64, max=2), T([1, 384, 1, 1], f32), T([1, 384, 1, 1], f32), T([384], f32), T([1, 384, 1, 1], f32), T([384], f32), T([], i64, max=2), T([1, 384, 1, 1], f32), T([1, 384, 1, 1], f32), T([384], f32), T([1, 384, 1, 1], f32), T([384], f32), T([], i64, max=2), T([1, 384, 1, 1], f32), T([1, 384, 1, 1], f32), T([384], f32), T([1, 384, 1, 1], f32), T([384], f32), T([], i64, max=2), T([1, 384, 1, 1], f32), T([1, 384, 1, 1], f32), T([384], f32), T([1, 384, 1, 1], f32), T([384], f32), T([], i64, max=2), T([1, 384, 1, 1], f32), T([1, 384, 1, 1], f32), T([384], f32), T([1, 384, 1, 1], f32), T([384], f32), T([], i64, max=2), T([1, 384, 1, 1], f32), T([1, 384, 1, 1], f32), T([384], f32), T([1, 384, 1, 1], f32), T([384], f32), T([], i64, max=2), T([1, 384, 1, 1], f32), T([1, 384, 1, 1], f32), T([384], f32), T([1, 384, 1, 1], f32), T([384], f32), T([], i64, max=2), T([1, 384, 1, 1], f32), T([1, 384, 1, 1], f32), T([384], f32), T([1, 384, 1, 1], f32), T([384], f32), T([], i64, max=2), T([1, 384, 1, 1], f32), T([1, 384, 1, 1], f32), T([384], f32), T([1, 384, 1, 1], f32), T([384], f32), T([], i64, max=2), T([1, 384, 1, 1], f32), T([1, 384, 1, 1], f32), T([384], f32), T([1, 384, 1, 1], f32), T([384], f32), T([], i64, max=2), T([1, 384, 1, 1], f32), T([1, 384, 1, 1], f32), T([384], f32), T([1, 384, 1, 1], f32), T([384], f32), T([], i64, max=2), T([1, 384, 1, 1], f32), T([1, 384, 1, 1], f32), T([384], f32), T([1, 384, 1, 1], f32), T([384], f32), T([], i64, max=2), T([1, 384, 1, 1], f32), T([1, 384, 1, 1], f32), T([384], f32), T([1, 384, 1, 1], f32), T([384], f32), T([], i64, max=2), T([1, 384, 1, 1], f32), T([1, 384, 1, 1], f32), T([384], f32), T([1, 384, 1, 1], f32), T([384], f32), T([], i64, max=2), T([1, 384, 1, 1], f32), T([1, 384, 1, 1], f32), T([384], f32), T([1, 384, 1, 1], f32), T([384], f32), T([], i64, max=2), T([1, 384, 1, 1], f32), T([1, 384, 1, 1], f32), T([384], f32), T([1, 384, 1, 1], f32), T([384], f32), T([], i64, max=2), T([1, 384, 1, 1], f32), T([1, 384, 1, 1], f32), T([384], f32), T([1, 384, 1, 1], f32), T([384], f32), T([], i64, max=2), T([1, 384, 1, 1], f32), T([1, 384, 1, 1], f32), T([384], f32), T([1, 384, 1, 1], f32), T([384], f32), T([], i64, max=2), T([1, 384, 1, 1], f32), T([1, 384, 1, 1], f32), T([384], f32), T([1, 384, 1, 1], f32), T([384], f32), T([], i64, max=2), T([1, 384, 1, 1], f32), T([1, 384, 1, 1], f32), T([384], f32), T([1, 384, 1, 1], f32), T([384], f32), T([], i64, max=2), T([1, 384, 1, 1], f32), T([1, 384, 1, 1], f32), T([384], f32), T([1, 384, 1, 1], f32), T([384], f32), T([], i64, max=2), T([1, 384, 1, 1], f32), T([1, 384, 1, 1], f32), T([384], f32), T([1, 384, 1, 1], f32), T([384], f32), T([], i64, max=2), T([1, 384, 1, 1], f32), T([1, 384, 1, 1], f32), T([384], f32), T([1, 384, 1, 1], f32), T([384], f32), T([], i64, max=2), T([1, 384, 1, 1], f32), T([1, 384, 1, 1], f32), T([384], f32), T([1, 384, 1, 1], f32), T([384], f32), T([], i64, max=2), T([1, 384, 1, 1], f32), T([1, 384, 1, 1], f32), T([384], f32), T([1, 384, 1, 1], f32), T([384], f32), T([], i64, max=2), T([1, 384, 1, 1], f32), T([1, 384, 1, 1], f32), T([384], f32), T([1, 384, 1, 1], f32), T([384], f32), T([], i64, max=2), T([1, 384, 1, 1], f32), T([1, 384, 1, 1], f32), T([384], f32), T([1, 384, 1, 1], f32), T([384], f32), T([], i64, max=2), T([1, 384, 1, 1], f32), T([1, 384, 1, 1], f32), T([384], f32), T([1, 384, 1, 1], f32), T([384], f32), T([], i64, max=2), T([1, 384, 1, 1], f32), T([1, 384, 1, 1], f32), T([384], f32), T([1, 384, 1, 1], f32), T([384], f32), T([], i64, max=2), T([1, 384, 1, 1], f32), T([1, 384, 1, 1], f32), T([384], f32), T([1, 384, 1, 1], f32), T([384], f32), T([], i64, max=2), T([1, 384, 1, 1], f32), T([1, 384, 1, 1], f32), T([384], f32), T([1, 384, 1, 1], f32), T([384], f32), T([], i64, max=2), T([1, 384, 1, 1], f32), T([1, 384, 1, 1], f32), T([384], f32), T([1, 384, 1, 1], f32), T([384], f32), T([], i64, max=2), T([1, 384, 1, 1], f32), T([1, 384, 1, 1], f32), T([384], f32), T([1, 384, 1, 1], f32), T([384], f32), T([], i64, max=2), T([1, 384, 1, 1], f32), T([1, 384, 1, 1], f32), T([384], f32), T([1, 384, 1, 1], f32), T([384], f32), T([], i64, max=2), T([1, 384, 1, 1], f32), T([1, 384, 1, 1], f32), T([384], f32), T([1, 384, 1, 1], f32), T([384], f32), T([], i64, max=2), T([1, 384, 1, 1], f32), T([1, 384, 1, 1], f32), T([384], f32), T([1, 384, 1, 1], f32), T([384], f32), T([], i64, max=2), T([1, 384, 1, 1], f32), T([1, 384, 1, 1], f32), T([384], f32), T([1, 384, 1, 1], f32), T([384], f32), T([], i64, max=2), T([8, 1408, 7, 7], f16), T([1408], f32), T([1408], f32), T([1408], f32), T([1408], f32), T([], i64, max=2), T([8, 1408, 7, 7], f16), T([1408], f32), T([1408], f32), T([1408], f32), T([1408], f32), T([1000], f32), T([1000, 1408], f32), S([8, 1408]))"

class Repro(torch.nn.Module):
    def forward(self, arg2_1: "i64[]", getitem_1: "f32[1, 64, 1, 1]", rsqrt: "f32[1, 64, 1, 1]", arg3_1: "f32[64]", getitem: "f32[1, 64, 1, 1]", arg4_1: "f32[64]", arg8_1: "i64[]", getitem_3: "f32[1, 64, 1, 1]", rsqrt_1: "f32[1, 64, 1, 1]", arg9_1: "f32[64]", getitem_2: "f32[1, 64, 1, 1]", arg10_1: "f32[64]", arg14_1: "i64[]", getitem_5: "f32[1, 96, 1, 1]", rsqrt_2: "f32[1, 96, 1, 1]", arg15_1: "f32[96]", getitem_4: "f32[1, 96, 1, 1]", arg16_1: "f32[96]", arg20_1: "i64[]", getitem_7: "f32[1, 96, 1, 1]", rsqrt_3: "f32[1, 96, 1, 1]", arg21_1: "f32[96]", getitem_6: "f32[1, 96, 1, 1]", arg22_1: "f32[96]", arg25_1: "i64[]", getitem_9: "f32[1, 96, 1, 1]", rsqrt_4: "f32[1, 96, 1, 1]", arg26_1: "f32[96]", getitem_8: "f32[1, 96, 1, 1]", arg27_1: "f32[96]", arg31_1: "i64[]", getitem_11: "f32[1, 96, 1, 1]", rsqrt_5: "f32[1, 96, 1, 1]", arg32_1: "f32[96]", getitem_10: "f32[1, 96, 1, 1]", arg33_1: "f32[96]", arg37_1: "i64[]", getitem_13: "f32[1, 96, 1, 1]", rsqrt_6: "f32[1, 96, 1, 1]", arg38_1: "f32[96]", getitem_12: "f32[1, 96, 1, 1]", arg39_1: "f32[96]", arg43_1: "i64[]", getitem_15: "f32[1, 192, 1, 1]", rsqrt_7: "f32[1, 192, 1, 1]", arg44_1: "f32[192]", getitem_14: "f32[1, 192, 1, 1]", arg45_1: "f32[192]", arg49_1: "i64[]", getitem_17: "f32[1, 192, 1, 1]", rsqrt_8: "f32[1, 192, 1, 1]", arg50_1: "f32[192]", getitem_16: "f32[1, 192, 1, 1]", arg51_1: "f32[192]", arg54_1: "i64[]", getitem_19: "f32[1, 192, 1, 1]", rsqrt_9: "f32[1, 192, 1, 1]", arg55_1: "f32[192]", getitem_18: "f32[1, 192, 1, 1]", arg56_1: "f32[192]", arg60_1: "i64[]", getitem_21: "f32[1, 192, 1, 1]", rsqrt_10: "f32[1, 192, 1, 1]", arg61_1: "f32[192]", getitem_20: "f32[1, 192, 1, 1]", arg62_1: "f32[192]", arg66_1: "i64[]", getitem_23: "f32[1, 192, 1, 1]", rsqrt_11: "f32[1, 192, 1, 1]", arg67_1: "f32[192]", getitem_22: "f32[1, 192, 1, 1]", arg68_1: "f32[192]", arg71_1: "i64[]", getitem_25: "f32[1, 192, 1, 1]", rsqrt_12: "f32[1, 192, 1, 1]", arg72_1: "f32[192]", getitem_24: "f32[1, 192, 1, 1]", arg73_1: "f32[192]", arg77_1: "i64[]", getitem_27: "f32[1, 192, 1, 1]", rsqrt_13: "f32[1, 192, 1, 1]", arg78_1: "f32[192]", getitem_26: "f32[1, 192, 1, 1]", arg79_1: "f32[192]", arg83_1: "i64[]", getitem_29: "f32[1, 192, 1, 1]", rsqrt_14: "f32[1, 192, 1, 1]", arg84_1: "f32[192]", getitem_28: "f32[1, 192, 1, 1]", arg85_1: "f32[192]", arg88_1: "i64[]", getitem_31: "f32[1, 192, 1, 1]", rsqrt_15: "f32[1, 192, 1, 1]", arg89_1: "f32[192]", getitem_30: "f32[1, 192, 1, 1]", arg90_1: "f32[192]", arg94_1: "i64[]", getitem_33: "f32[1, 192, 1, 1]", rsqrt_16: "f32[1, 192, 1, 1]", arg95_1: "f32[192]", getitem_32: "f32[1, 192, 1, 1]", arg96_1: "f32[192]", arg100_1: "i64[]", getitem_35: "f32[1, 192, 1, 1]", rsqrt_17: "f32[1, 192, 1, 1]", arg101_1: "f32[192]", getitem_34: "f32[1, 192, 1, 1]", arg102_1: "f32[192]", arg106_1: "i64[]", getitem_37: "f32[1, 384, 1, 1]", rsqrt_18: "f32[1, 384, 1, 1]", arg107_1: "f32[384]", getitem_36: "f32[1, 384, 1, 1]", arg108_1: "f32[384]", arg112_1: "i64[]", getitem_39: "f32[1, 384, 1, 1]", rsqrt_19: "f32[1, 384, 1, 1]", arg113_1: "f32[384]", getitem_38: "f32[1, 384, 1, 1]", arg114_1: "f32[384]", arg117_1: "i64[]", getitem_41: "f32[1, 384, 1, 1]", rsqrt_20: "f32[1, 384, 1, 1]", arg118_1: "f32[384]", getitem_40: "f32[1, 384, 1, 1]", arg119_1: "f32[384]", arg123_1: "i64[]", getitem_43: "f32[1, 384, 1, 1]", rsqrt_21: "f32[1, 384, 1, 1]", arg124_1: "f32[384]", getitem_42: "f32[1, 384, 1, 1]", arg125_1: "f32[384]", arg129_1: "i64[]", getitem_45: "f32[1, 384, 1, 1]", rsqrt_22: "f32[1, 384, 1, 1]", arg130_1: "f32[384]", getitem_44: "f32[1, 384, 1, 1]", arg131_1: "f32[384]", arg134_1: "i64[]", getitem_47: "f32[1, 384, 1, 1]", rsqrt_23: "f32[1, 384, 1, 1]", arg135_1: "f32[384]", getitem_46: "f32[1, 384, 1, 1]", arg136_1: "f32[384]", arg140_1: "i64[]", getitem_49: "f32[1, 384, 1, 1]", rsqrt_24: "f32[1, 384, 1, 1]", arg141_1: "f32[384]", getitem_48: "f32[1, 384, 1, 1]", arg142_1: "f32[384]", arg146_1: "i64[]", getitem_51: "f32[1, 384, 1, 1]", rsqrt_25: "f32[1, 384, 1, 1]", arg147_1: "f32[384]", getitem_50: "f32[1, 384, 1, 1]", arg148_1: "f32[384]", arg151_1: "i64[]", getitem_53: "f32[1, 384, 1, 1]", rsqrt_26: "f32[1, 384, 1, 1]", arg152_1: "f32[384]", getitem_52: "f32[1, 384, 1, 1]", arg153_1: "f32[384]", arg157_1: "i64[]", getitem_55: "f32[1, 384, 1, 1]", rsqrt_27: "f32[1, 384, 1, 1]", arg158_1: "f32[384]", getitem_54: "f32[1, 384, 1, 1]", arg159_1: "f32[384]", arg163_1: "i64[]", getitem_57: "f32[1, 384, 1, 1]", rsqrt_28: "f32[1, 384, 1, 1]", arg164_1: "f32[384]", getitem_56: "f32[1, 384, 1, 1]", arg165_1: "f32[384]", arg168_1: "i64[]", getitem_59: "f32[1, 384, 1, 1]", rsqrt_29: "f32[1, 384, 1, 1]", arg169_1: "f32[384]", getitem_58: "f32[1, 384, 1, 1]", arg170_1: "f32[384]", arg174_1: "i64[]", getitem_61: "f32[1, 384, 1, 1]", rsqrt_30: "f32[1, 384, 1, 1]", arg175_1: "f32[384]", getitem_60: "f32[1, 384, 1, 1]", arg176_1: "f32[384]", arg180_1: "i64[]", getitem_63: "f32[1, 384, 1, 1]", rsqrt_31: "f32[1, 384, 1, 1]", arg181_1: "f32[384]", getitem_62: "f32[1, 384, 1, 1]", arg182_1: "f32[384]", arg185_1: "i64[]", getitem_65: "f32[1, 384, 1, 1]", rsqrt_32: "f32[1, 384, 1, 1]", arg186_1: "f32[384]", getitem_64: "f32[1, 384, 1, 1]", arg187_1: "f32[384]", arg191_1: "i64[]", getitem_67: "f32[1, 384, 1, 1]", rsqrt_33: "f32[1, 384, 1, 1]", arg192_1: "f32[384]", getitem_66: "f32[1, 384, 1, 1]", arg193_1: "f32[384]", arg197_1: "i64[]", getitem_69: "f32[1, 384, 1, 1]", rsqrt_34: "f32[1, 384, 1, 1]", arg198_1: "f32[384]", getitem_68: "f32[1, 384, 1, 1]", arg199_1: "f32[384]", arg202_1: "i64[]", getitem_71: "f32[1, 384, 1, 1]", rsqrt_35: "f32[1, 384, 1, 1]", arg203_1: "f32[384]", getitem_70: "f32[1, 384, 1, 1]", arg204_1: "f32[384]", arg208_1: "i64[]", getitem_73: "f32[1, 384, 1, 1]", rsqrt_36: "f32[1, 384, 1, 1]", arg209_1: "f32[384]", getitem_72: "f32[1, 384, 1, 1]", arg210_1: "f32[384]", arg214_1: "i64[]", getitem_75: "f32[1, 384, 1, 1]", rsqrt_37: "f32[1, 384, 1, 1]", arg215_1: "f32[384]", getitem_74: "f32[1, 384, 1, 1]", arg216_1: "f32[384]", arg219_1: "i64[]", getitem_77: "f32[1, 384, 1, 1]", rsqrt_38: "f32[1, 384, 1, 1]", arg220_1: "f32[384]", getitem_76: "f32[1, 384, 1, 1]", arg221_1: "f32[384]", arg225_1: "i64[]", getitem_79: "f32[1, 384, 1, 1]", rsqrt_39: "f32[1, 384, 1, 1]", arg226_1: "f32[384]", getitem_78: "f32[1, 384, 1, 1]", arg227_1: "f32[384]", arg231_1: "i64[]", getitem_81: "f32[1, 384, 1, 1]", rsqrt_40: "f32[1, 384, 1, 1]", arg232_1: "f32[384]", getitem_80: "f32[1, 384, 1, 1]", arg233_1: "f32[384]", arg236_1: "i64[]", getitem_83: "f32[1, 384, 1, 1]", rsqrt_41: "f32[1, 384, 1, 1]", arg237_1: "f32[384]", getitem_82: "f32[1, 384, 1, 1]", arg238_1: "f32[384]", arg242_1: "i64[]", getitem_85: "f32[1, 384, 1, 1]", rsqrt_42: "f32[1, 384, 1, 1]", arg243_1: "f32[384]", getitem_84: "f32[1, 384, 1, 1]", arg244_1: "f32[384]", arg248_1: "i64[]", getitem_87: "f32[1, 384, 1, 1]", rsqrt_43: "f32[1, 384, 1, 1]", arg249_1: "f32[384]", getitem_86: "f32[1, 384, 1, 1]", arg250_1: "f32[384]", arg253_1: "i64[]", getitem_89: "f32[1, 384, 1, 1]", rsqrt_44: "f32[1, 384, 1, 1]", arg254_1: "f32[384]", getitem_88: "f32[1, 384, 1, 1]", arg255_1: "f32[384]", arg259_1: "i64[]", getitem_91: "f32[1, 384, 1, 1]", rsqrt_45: "f32[1, 384, 1, 1]", arg260_1: "f32[384]", getitem_90: "f32[1, 384, 1, 1]", arg261_1: "f32[384]", arg265_1: "i64[]", getitem_93: "f32[1, 384, 1, 1]", rsqrt_46: "f32[1, 384, 1, 1]", arg266_1: "f32[384]", getitem_92: "f32[1, 384, 1, 1]", arg267_1: "f32[384]", arg270_1: "i64[]", getitem_95: "f32[1, 384, 1, 1]", rsqrt_47: "f32[1, 384, 1, 1]", arg271_1: "f32[384]", getitem_94: "f32[1, 384, 1, 1]", arg272_1: "f32[384]", arg276_1: "i64[]", getitem_97: "f32[1, 384, 1, 1]", rsqrt_48: "f32[1, 384, 1, 1]", arg277_1: "f32[384]", getitem_96: "f32[1, 384, 1, 1]", arg278_1: "f32[384]", arg282_1: "i64[]", getitem_99: "f32[1, 384, 1, 1]", rsqrt_49: "f32[1, 384, 1, 1]", arg283_1: "f32[384]", getitem_98: "f32[1, 384, 1, 1]", arg284_1: "f32[384]", arg287_1: "i64[]", getitem_101: "f32[1, 384, 1, 1]", rsqrt_50: "f32[1, 384, 1, 1]", arg288_1: "f32[384]", getitem_100: "f32[1, 384, 1, 1]", arg289_1: "f32[384]", arg293_1: "i64[]", getitem_103: "f32[1, 384, 1, 1]", rsqrt_51: "f32[1, 384, 1, 1]", arg294_1: "f32[384]", getitem_102: "f32[1, 384, 1, 1]", arg295_1: "f32[384]", arg299_1: "i64[]", getitem_105: "f32[1, 384, 1, 1]", rsqrt_52: "f32[1, 384, 1, 1]", arg300_1: "f32[384]", getitem_104: "f32[1, 384, 1, 1]", arg301_1: "f32[384]", arg304_1: "i64[]", getitem_107: "f32[1, 384, 1, 1]", rsqrt_53: "f32[1, 384, 1, 1]", arg305_1: "f32[384]", getitem_106: "f32[1, 384, 1, 1]", arg306_1: "f32[384]", arg310_1: "i64[]", getitem_109: "f32[1, 384, 1, 1]", rsqrt_54: "f32[1, 384, 1, 1]", arg311_1: "f32[384]", getitem_108: "f32[1, 384, 1, 1]", arg312_1: "f32[384]", arg316_1: "i64[]", getitem_111: "f32[1, 384, 1, 1]", rsqrt_55: "f32[1, 384, 1, 1]", arg317_1: "f32[384]", getitem_110: "f32[1, 384, 1, 1]", arg318_1: "f32[384]", arg321_1: "i64[]", getitem_113: "f32[1, 384, 1, 1]", rsqrt_56: "f32[1, 384, 1, 1]", arg322_1: "f32[384]", getitem_112: "f32[1, 384, 1, 1]", arg323_1: "f32[384]", arg327_1: "i64[]", getitem_115: "f32[1, 384, 1, 1]", rsqrt_57: "f32[1, 384, 1, 1]", arg328_1: "f32[384]", getitem_114: "f32[1, 384, 1, 1]", arg329_1: "f32[384]", arg333_1: "i64[]", getitem_117: "f32[1, 384, 1, 1]", rsqrt_58: "f32[1, 384, 1, 1]", arg334_1: "f32[384]", getitem_116: "f32[1, 384, 1, 1]", arg335_1: "f32[384]", arg339_1: "i64[]", convolution_42: "f16[8, 1408, 7, 7]", arg340_1: "f32[1408]", arg341_1: "f32[1408]", arg342_1: "f32[1408]", arg343_1: "f32[1408]", arg345_1: "i64[]", convolution_43: "f16[8, 1408, 7, 7]", arg346_1: "f32[1408]", arg347_1: "f32[1408]", arg348_1: "f32[1408]", arg349_1: "f32[1408]", arg351_1: "f32[1000]", arg350_1: "f32[1000, 1408]", _shape_param_0):
        # No stacktrace found for following nodes
        add_tensor: "i64[]" = torch.ops.aten.add.Tensor(arg2_1, 1)
        squeeze_dims: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        squeeze_dims_1: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
        mul_tensor: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims, 0.1)
        mul_tensor_1: "f32[64]" = torch.ops.aten.mul.Tensor(arg3_1, 0.9)
        add_tensor_1: "f32[64]" = torch.ops.aten.add.Tensor(mul_tensor, mul_tensor_1);  mul_tensor = mul_tensor_1 = None
        squeeze_dims_2: "f32[64]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_tensor_2: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims_2, 1.00000996502277);  squeeze_dims_2 = None
        mul_tensor_3: "f32[64]" = torch.ops.aten.mul.Tensor(mul_tensor_2, 0.1);  mul_tensor_2 = None
        mul_tensor_4: "f32[64]" = torch.ops.aten.mul.Tensor(arg4_1, 0.9)
        add_tensor_2: "f32[64]" = torch.ops.aten.add.Tensor(mul_tensor_3, mul_tensor_4);  mul_tensor_3 = mul_tensor_4 = None
        add_tensor_3: "i64[]" = torch.ops.aten.add.Tensor(arg8_1, 1)
        squeeze_dims_3: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
        squeeze_dims_4: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
        mul_tensor_5: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims_3, 0.1)
        mul_tensor_6: "f32[64]" = torch.ops.aten.mul.Tensor(arg9_1, 0.9)
        add_tensor_4: "f32[64]" = torch.ops.aten.add.Tensor(mul_tensor_5, mul_tensor_6);  mul_tensor_5 = mul_tensor_6 = None
        squeeze_dims_5: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
        mul_tensor_7: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims_5, 1.00000996502277);  squeeze_dims_5 = None
        mul_tensor_8: "f32[64]" = torch.ops.aten.mul.Tensor(mul_tensor_7, 0.1);  mul_tensor_7 = None
        mul_tensor_9: "f32[64]" = torch.ops.aten.mul.Tensor(arg10_1, 0.9)
        add_tensor_5: "f32[64]" = torch.ops.aten.add.Tensor(mul_tensor_8, mul_tensor_9);  mul_tensor_8 = mul_tensor_9 = None
        add_tensor_6: "i64[]" = torch.ops.aten.add.Tensor(arg14_1, 1)
        squeeze_dims_6: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
        squeeze_dims_7: "f32[96]" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
        mul_tensor_10: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_dims_6, 0.1)
        mul_tensor_11: "f32[96]" = torch.ops.aten.mul.Tensor(arg15_1, 0.9)
        add_tensor_7: "f32[96]" = torch.ops.aten.add.Tensor(mul_tensor_10, mul_tensor_11);  mul_tensor_10 = mul_tensor_11 = None
        squeeze_dims_8: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
        mul_tensor_12: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_dims_8, 1.0000398612827361);  squeeze_dims_8 = None
        mul_tensor_13: "f32[96]" = torch.ops.aten.mul.Tensor(mul_tensor_12, 0.1);  mul_tensor_12 = None
        mul_tensor_14: "f32[96]" = torch.ops.aten.mul.Tensor(arg16_1, 0.9)
        add_tensor_8: "f32[96]" = torch.ops.aten.add.Tensor(mul_tensor_13, mul_tensor_14);  mul_tensor_13 = mul_tensor_14 = None
        add_tensor_9: "i64[]" = torch.ops.aten.add.Tensor(arg20_1, 1)
        squeeze_dims_9: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2, 3]);  getitem_7 = None
        squeeze_dims_10: "f32[96]" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
        mul_tensor_15: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_dims_9, 0.1)
        mul_tensor_16: "f32[96]" = torch.ops.aten.mul.Tensor(arg21_1, 0.9)
        add_tensor_10: "f32[96]" = torch.ops.aten.add.Tensor(mul_tensor_15, mul_tensor_16);  mul_tensor_15 = mul_tensor_16 = None
        squeeze_dims_11: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_6, [0, 2, 3]);  getitem_6 = None
        mul_tensor_17: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_dims_11, 1.0000398612827361);  squeeze_dims_11 = None
        mul_tensor_18: "f32[96]" = torch.ops.aten.mul.Tensor(mul_tensor_17, 0.1);  mul_tensor_17 = None
        mul_tensor_19: "f32[96]" = torch.ops.aten.mul.Tensor(arg22_1, 0.9)
        add_tensor_11: "f32[96]" = torch.ops.aten.add.Tensor(mul_tensor_18, mul_tensor_19);  mul_tensor_18 = mul_tensor_19 = None
        add_tensor_12: "i64[]" = torch.ops.aten.add.Tensor(arg25_1, 1)
        squeeze_dims_12: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3]);  getitem_9 = None
        squeeze_dims_13: "f32[96]" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
        mul_tensor_20: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_dims_12, 0.1)
        mul_tensor_21: "f32[96]" = torch.ops.aten.mul.Tensor(arg26_1, 0.9)
        add_tensor_13: "f32[96]" = torch.ops.aten.add.Tensor(mul_tensor_20, mul_tensor_21);  mul_tensor_20 = mul_tensor_21 = None
        squeeze_dims_14: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_8, [0, 2, 3]);  getitem_8 = None
        mul_tensor_22: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_dims_14, 1.0000398612827361);  squeeze_dims_14 = None
        mul_tensor_23: "f32[96]" = torch.ops.aten.mul.Tensor(mul_tensor_22, 0.1);  mul_tensor_22 = None
        mul_tensor_24: "f32[96]" = torch.ops.aten.mul.Tensor(arg27_1, 0.9)
        add_tensor_14: "f32[96]" = torch.ops.aten.add.Tensor(mul_tensor_23, mul_tensor_24);  mul_tensor_23 = mul_tensor_24 = None
        add_tensor_15: "i64[]" = torch.ops.aten.add.Tensor(arg31_1, 1)
        squeeze_dims_15: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3]);  getitem_11 = None
        squeeze_dims_16: "f32[96]" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
        mul_tensor_25: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_dims_15, 0.1)
        mul_tensor_26: "f32[96]" = torch.ops.aten.mul.Tensor(arg32_1, 0.9)
        add_tensor_16: "f32[96]" = torch.ops.aten.add.Tensor(mul_tensor_25, mul_tensor_26);  mul_tensor_25 = mul_tensor_26 = None
        squeeze_dims_17: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_10, [0, 2, 3]);  getitem_10 = None
        mul_tensor_27: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_dims_17, 1.0000398612827361);  squeeze_dims_17 = None
        mul_tensor_28: "f32[96]" = torch.ops.aten.mul.Tensor(mul_tensor_27, 0.1);  mul_tensor_27 = None
        mul_tensor_29: "f32[96]" = torch.ops.aten.mul.Tensor(arg33_1, 0.9)
        add_tensor_17: "f32[96]" = torch.ops.aten.add.Tensor(mul_tensor_28, mul_tensor_29);  mul_tensor_28 = mul_tensor_29 = None
        add_tensor_18: "i64[]" = torch.ops.aten.add.Tensor(arg37_1, 1)
        squeeze_dims_18: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
        squeeze_dims_19: "f32[96]" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
        mul_tensor_30: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_dims_18, 0.1)
        mul_tensor_31: "f32[96]" = torch.ops.aten.mul.Tensor(arg38_1, 0.9)
        add_tensor_19: "f32[96]" = torch.ops.aten.add.Tensor(mul_tensor_30, mul_tensor_31);  mul_tensor_30 = mul_tensor_31 = None
        squeeze_dims_20: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_12, [0, 2, 3]);  getitem_12 = None
        mul_tensor_32: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_dims_20, 1.0000398612827361);  squeeze_dims_20 = None
        mul_tensor_33: "f32[96]" = torch.ops.aten.mul.Tensor(mul_tensor_32, 0.1);  mul_tensor_32 = None
        mul_tensor_34: "f32[96]" = torch.ops.aten.mul.Tensor(arg39_1, 0.9)
        add_tensor_20: "f32[96]" = torch.ops.aten.add.Tensor(mul_tensor_33, mul_tensor_34);  mul_tensor_33 = mul_tensor_34 = None
        add_tensor_21: "i64[]" = torch.ops.aten.add.Tensor(arg43_1, 1)
        squeeze_dims_21: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3]);  getitem_15 = None
        squeeze_dims_22: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2, 3]);  rsqrt_7 = None
        mul_tensor_35: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_21, 0.1)
        mul_tensor_36: "f32[192]" = torch.ops.aten.mul.Tensor(arg44_1, 0.9)
        add_tensor_22: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_35, mul_tensor_36);  mul_tensor_35 = mul_tensor_36 = None
        squeeze_dims_23: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_14, [0, 2, 3]);  getitem_14 = None
        mul_tensor_37: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_23, 1.0001594642002871);  squeeze_dims_23 = None
        mul_tensor_38: "f32[192]" = torch.ops.aten.mul.Tensor(mul_tensor_37, 0.1);  mul_tensor_37 = None
        mul_tensor_39: "f32[192]" = torch.ops.aten.mul.Tensor(arg45_1, 0.9)
        add_tensor_23: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_38, mul_tensor_39);  mul_tensor_38 = mul_tensor_39 = None
        add_tensor_24: "i64[]" = torch.ops.aten.add.Tensor(arg49_1, 1)
        squeeze_dims_24: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_17, [0, 2, 3]);  getitem_17 = None
        squeeze_dims_25: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2, 3]);  rsqrt_8 = None
        mul_tensor_40: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_24, 0.1)
        mul_tensor_41: "f32[192]" = torch.ops.aten.mul.Tensor(arg50_1, 0.9)
        add_tensor_25: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_40, mul_tensor_41);  mul_tensor_40 = mul_tensor_41 = None
        squeeze_dims_26: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_16, [0, 2, 3]);  getitem_16 = None
        mul_tensor_42: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_26, 1.0001594642002871);  squeeze_dims_26 = None
        mul_tensor_43: "f32[192]" = torch.ops.aten.mul.Tensor(mul_tensor_42, 0.1);  mul_tensor_42 = None
        mul_tensor_44: "f32[192]" = torch.ops.aten.mul.Tensor(arg51_1, 0.9)
        add_tensor_26: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_43, mul_tensor_44);  mul_tensor_43 = mul_tensor_44 = None
        add_tensor_27: "i64[]" = torch.ops.aten.add.Tensor(arg54_1, 1)
        squeeze_dims_27: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3]);  getitem_19 = None
        squeeze_dims_28: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
        mul_tensor_45: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_27, 0.1)
        mul_tensor_46: "f32[192]" = torch.ops.aten.mul.Tensor(arg55_1, 0.9)
        add_tensor_28: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_45, mul_tensor_46);  mul_tensor_45 = mul_tensor_46 = None
        squeeze_dims_29: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_18, [0, 2, 3]);  getitem_18 = None
        mul_tensor_47: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_29, 1.0001594642002871);  squeeze_dims_29 = None
        mul_tensor_48: "f32[192]" = torch.ops.aten.mul.Tensor(mul_tensor_47, 0.1);  mul_tensor_47 = None
        mul_tensor_49: "f32[192]" = torch.ops.aten.mul.Tensor(arg56_1, 0.9)
        add_tensor_29: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_48, mul_tensor_49);  mul_tensor_48 = mul_tensor_49 = None
        add_tensor_30: "i64[]" = torch.ops.aten.add.Tensor(arg60_1, 1)
        squeeze_dims_30: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
        squeeze_dims_31: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
        mul_tensor_50: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_30, 0.1)
        mul_tensor_51: "f32[192]" = torch.ops.aten.mul.Tensor(arg61_1, 0.9)
        add_tensor_31: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_50, mul_tensor_51);  mul_tensor_50 = mul_tensor_51 = None
        squeeze_dims_32: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_20, [0, 2, 3]);  getitem_20 = None
        mul_tensor_52: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_32, 1.0001594642002871);  squeeze_dims_32 = None
        mul_tensor_53: "f32[192]" = torch.ops.aten.mul.Tensor(mul_tensor_52, 0.1);  mul_tensor_52 = None
        mul_tensor_54: "f32[192]" = torch.ops.aten.mul.Tensor(arg62_1, 0.9)
        add_tensor_32: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_53, mul_tensor_54);  mul_tensor_53 = mul_tensor_54 = None
        add_tensor_33: "i64[]" = torch.ops.aten.add.Tensor(arg66_1, 1)
        squeeze_dims_33: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_23, [0, 2, 3]);  getitem_23 = None
        squeeze_dims_34: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2, 3]);  rsqrt_11 = None
        mul_tensor_55: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_33, 0.1)
        mul_tensor_56: "f32[192]" = torch.ops.aten.mul.Tensor(arg67_1, 0.9)
        add_tensor_34: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_55, mul_tensor_56);  mul_tensor_55 = mul_tensor_56 = None
        squeeze_dims_35: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_22, [0, 2, 3]);  getitem_22 = None
        mul_tensor_57: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_35, 1.0001594642002871);  squeeze_dims_35 = None
        mul_tensor_58: "f32[192]" = torch.ops.aten.mul.Tensor(mul_tensor_57, 0.1);  mul_tensor_57 = None
        mul_tensor_59: "f32[192]" = torch.ops.aten.mul.Tensor(arg68_1, 0.9)
        add_tensor_35: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_58, mul_tensor_59);  mul_tensor_58 = mul_tensor_59 = None
        add_tensor_36: "i64[]" = torch.ops.aten.add.Tensor(arg71_1, 1)
        squeeze_dims_36: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_25, [0, 2, 3]);  getitem_25 = None
        squeeze_dims_37: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
        mul_tensor_60: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_36, 0.1)
        mul_tensor_61: "f32[192]" = torch.ops.aten.mul.Tensor(arg72_1, 0.9)
        add_tensor_37: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_60, mul_tensor_61);  mul_tensor_60 = mul_tensor_61 = None
        squeeze_dims_38: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_24, [0, 2, 3]);  getitem_24 = None
        mul_tensor_62: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_38, 1.0001594642002871);  squeeze_dims_38 = None
        mul_tensor_63: "f32[192]" = torch.ops.aten.mul.Tensor(mul_tensor_62, 0.1);  mul_tensor_62 = None
        mul_tensor_64: "f32[192]" = torch.ops.aten.mul.Tensor(arg73_1, 0.9)
        add_tensor_38: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_63, mul_tensor_64);  mul_tensor_63 = mul_tensor_64 = None
        add_tensor_39: "i64[]" = torch.ops.aten.add.Tensor(arg77_1, 1)
        squeeze_dims_39: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3]);  getitem_27 = None
        squeeze_dims_40: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2, 3]);  rsqrt_13 = None
        mul_tensor_65: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_39, 0.1)
        mul_tensor_66: "f32[192]" = torch.ops.aten.mul.Tensor(arg78_1, 0.9)
        add_tensor_40: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_65, mul_tensor_66);  mul_tensor_65 = mul_tensor_66 = None
        squeeze_dims_41: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_26, [0, 2, 3]);  getitem_26 = None
        mul_tensor_67: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_41, 1.0001594642002871);  squeeze_dims_41 = None
        mul_tensor_68: "f32[192]" = torch.ops.aten.mul.Tensor(mul_tensor_67, 0.1);  mul_tensor_67 = None
        mul_tensor_69: "f32[192]" = torch.ops.aten.mul.Tensor(arg79_1, 0.9)
        add_tensor_41: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_68, mul_tensor_69);  mul_tensor_68 = mul_tensor_69 = None
        add_tensor_42: "i64[]" = torch.ops.aten.add.Tensor(arg83_1, 1)
        squeeze_dims_42: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_29, [0, 2, 3]);  getitem_29 = None
        squeeze_dims_43: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2, 3]);  rsqrt_14 = None
        mul_tensor_70: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_42, 0.1)
        mul_tensor_71: "f32[192]" = torch.ops.aten.mul.Tensor(arg84_1, 0.9)
        add_tensor_43: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_70, mul_tensor_71);  mul_tensor_70 = mul_tensor_71 = None
        squeeze_dims_44: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_28, [0, 2, 3]);  getitem_28 = None
        mul_tensor_72: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_44, 1.0001594642002871);  squeeze_dims_44 = None
        mul_tensor_73: "f32[192]" = torch.ops.aten.mul.Tensor(mul_tensor_72, 0.1);  mul_tensor_72 = None
        mul_tensor_74: "f32[192]" = torch.ops.aten.mul.Tensor(arg85_1, 0.9)
        add_tensor_44: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_73, mul_tensor_74);  mul_tensor_73 = mul_tensor_74 = None
        add_tensor_45: "i64[]" = torch.ops.aten.add.Tensor(arg88_1, 1)
        squeeze_dims_45: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2, 3]);  getitem_31 = None
        squeeze_dims_46: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2, 3]);  rsqrt_15 = None
        mul_tensor_75: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_45, 0.1)
        mul_tensor_76: "f32[192]" = torch.ops.aten.mul.Tensor(arg89_1, 0.9)
        add_tensor_46: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_75, mul_tensor_76);  mul_tensor_75 = mul_tensor_76 = None
        squeeze_dims_47: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_30, [0, 2, 3]);  getitem_30 = None
        mul_tensor_77: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_47, 1.0001594642002871);  squeeze_dims_47 = None
        mul_tensor_78: "f32[192]" = torch.ops.aten.mul.Tensor(mul_tensor_77, 0.1);  mul_tensor_77 = None
        mul_tensor_79: "f32[192]" = torch.ops.aten.mul.Tensor(arg90_1, 0.9)
        add_tensor_47: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_78, mul_tensor_79);  mul_tensor_78 = mul_tensor_79 = None
        add_tensor_48: "i64[]" = torch.ops.aten.add.Tensor(arg94_1, 1)
        squeeze_dims_48: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3]);  getitem_33 = None
        squeeze_dims_49: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2, 3]);  rsqrt_16 = None
        mul_tensor_80: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_48, 0.1)
        mul_tensor_81: "f32[192]" = torch.ops.aten.mul.Tensor(arg95_1, 0.9)
        add_tensor_49: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_80, mul_tensor_81);  mul_tensor_80 = mul_tensor_81 = None
        squeeze_dims_50: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_32, [0, 2, 3]);  getitem_32 = None
        mul_tensor_82: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_50, 1.0001594642002871);  squeeze_dims_50 = None
        mul_tensor_83: "f32[192]" = torch.ops.aten.mul.Tensor(mul_tensor_82, 0.1);  mul_tensor_82 = None
        mul_tensor_84: "f32[192]" = torch.ops.aten.mul.Tensor(arg96_1, 0.9)
        add_tensor_50: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_83, mul_tensor_84);  mul_tensor_83 = mul_tensor_84 = None
        add_tensor_51: "i64[]" = torch.ops.aten.add.Tensor(arg100_1, 1)
        squeeze_dims_51: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_35, [0, 2, 3]);  getitem_35 = None
        squeeze_dims_52: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_17, [0, 2, 3]);  rsqrt_17 = None
        mul_tensor_85: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_51, 0.1)
        mul_tensor_86: "f32[192]" = torch.ops.aten.mul.Tensor(arg101_1, 0.9)
        add_tensor_52: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_85, mul_tensor_86);  mul_tensor_85 = mul_tensor_86 = None
        squeeze_dims_53: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_34, [0, 2, 3]);  getitem_34 = None
        mul_tensor_87: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_53, 1.0001594642002871);  squeeze_dims_53 = None
        mul_tensor_88: "f32[192]" = torch.ops.aten.mul.Tensor(mul_tensor_87, 0.1);  mul_tensor_87 = None
        mul_tensor_89: "f32[192]" = torch.ops.aten.mul.Tensor(arg102_1, 0.9)
        add_tensor_53: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_88, mul_tensor_89);  mul_tensor_88 = mul_tensor_89 = None
        add_tensor_54: "i64[]" = torch.ops.aten.add.Tensor(arg106_1, 1)
        squeeze_dims_54: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_37, [0, 2, 3]);  getitem_37 = None
        squeeze_dims_55: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_18, [0, 2, 3]);  rsqrt_18 = None
        mul_tensor_90: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_54, 0.1)
        mul_tensor_91: "f32[384]" = torch.ops.aten.mul.Tensor(arg107_1, 0.9)
        add_tensor_55: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_90, mul_tensor_91);  mul_tensor_90 = mul_tensor_91 = None
        squeeze_dims_56: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_36, [0, 2, 3]);  getitem_36 = None
        mul_tensor_92: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_56, 1.0006381620931717);  squeeze_dims_56 = None
        mul_tensor_93: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_92, 0.1);  mul_tensor_92 = None
        mul_tensor_94: "f32[384]" = torch.ops.aten.mul.Tensor(arg108_1, 0.9)
        add_tensor_56: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_93, mul_tensor_94);  mul_tensor_93 = mul_tensor_94 = None
        add_tensor_57: "i64[]" = torch.ops.aten.add.Tensor(arg112_1, 1)
        squeeze_dims_57: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_39, [0, 2, 3]);  getitem_39 = None
        squeeze_dims_58: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_19, [0, 2, 3]);  rsqrt_19 = None
        mul_tensor_95: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_57, 0.1)
        mul_tensor_96: "f32[384]" = torch.ops.aten.mul.Tensor(arg113_1, 0.9)
        add_tensor_58: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_95, mul_tensor_96);  mul_tensor_95 = mul_tensor_96 = None
        squeeze_dims_59: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_38, [0, 2, 3]);  getitem_38 = None
        mul_tensor_97: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_59, 1.0006381620931717);  squeeze_dims_59 = None
        mul_tensor_98: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_97, 0.1);  mul_tensor_97 = None
        mul_tensor_99: "f32[384]" = torch.ops.aten.mul.Tensor(arg114_1, 0.9)
        add_tensor_59: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_98, mul_tensor_99);  mul_tensor_98 = mul_tensor_99 = None
        add_tensor_60: "i64[]" = torch.ops.aten.add.Tensor(arg117_1, 1)
        squeeze_dims_60: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_41, [0, 2, 3]);  getitem_41 = None
        squeeze_dims_61: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_20, [0, 2, 3]);  rsqrt_20 = None
        mul_tensor_100: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_60, 0.1)
        mul_tensor_101: "f32[384]" = torch.ops.aten.mul.Tensor(arg118_1, 0.9)
        add_tensor_61: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_100, mul_tensor_101);  mul_tensor_100 = mul_tensor_101 = None
        squeeze_dims_62: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_40, [0, 2, 3]);  getitem_40 = None
        mul_tensor_102: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_62, 1.0006381620931717);  squeeze_dims_62 = None
        mul_tensor_103: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_102, 0.1);  mul_tensor_102 = None
        mul_tensor_104: "f32[384]" = torch.ops.aten.mul.Tensor(arg119_1, 0.9)
        add_tensor_62: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_103, mul_tensor_104);  mul_tensor_103 = mul_tensor_104 = None
        add_tensor_63: "i64[]" = torch.ops.aten.add.Tensor(arg123_1, 1)
        squeeze_dims_63: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_43, [0, 2, 3]);  getitem_43 = None
        squeeze_dims_64: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_21, [0, 2, 3]);  rsqrt_21 = None
        mul_tensor_105: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_63, 0.1)
        mul_tensor_106: "f32[384]" = torch.ops.aten.mul.Tensor(arg124_1, 0.9)
        add_tensor_64: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_105, mul_tensor_106);  mul_tensor_105 = mul_tensor_106 = None
        squeeze_dims_65: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_42, [0, 2, 3]);  getitem_42 = None
        mul_tensor_107: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_65, 1.0006381620931717);  squeeze_dims_65 = None
        mul_tensor_108: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_107, 0.1);  mul_tensor_107 = None
        mul_tensor_109: "f32[384]" = torch.ops.aten.mul.Tensor(arg125_1, 0.9)
        add_tensor_65: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_108, mul_tensor_109);  mul_tensor_108 = mul_tensor_109 = None
        add_tensor_66: "i64[]" = torch.ops.aten.add.Tensor(arg129_1, 1)
        squeeze_dims_66: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_45, [0, 2, 3]);  getitem_45 = None
        squeeze_dims_67: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_22, [0, 2, 3]);  rsqrt_22 = None
        mul_tensor_110: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_66, 0.1)
        mul_tensor_111: "f32[384]" = torch.ops.aten.mul.Tensor(arg130_1, 0.9)
        add_tensor_67: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_110, mul_tensor_111);  mul_tensor_110 = mul_tensor_111 = None
        squeeze_dims_68: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_44, [0, 2, 3]);  getitem_44 = None
        mul_tensor_112: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_68, 1.0006381620931717);  squeeze_dims_68 = None
        mul_tensor_113: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_112, 0.1);  mul_tensor_112 = None
        mul_tensor_114: "f32[384]" = torch.ops.aten.mul.Tensor(arg131_1, 0.9)
        add_tensor_68: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_113, mul_tensor_114);  mul_tensor_113 = mul_tensor_114 = None
        add_tensor_69: "i64[]" = torch.ops.aten.add.Tensor(arg134_1, 1)
        squeeze_dims_69: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_47, [0, 2, 3]);  getitem_47 = None
        squeeze_dims_70: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_23, [0, 2, 3]);  rsqrt_23 = None
        mul_tensor_115: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_69, 0.1)
        mul_tensor_116: "f32[384]" = torch.ops.aten.mul.Tensor(arg135_1, 0.9)
        add_tensor_70: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_115, mul_tensor_116);  mul_tensor_115 = mul_tensor_116 = None
        squeeze_dims_71: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_46, [0, 2, 3]);  getitem_46 = None
        mul_tensor_117: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_71, 1.0006381620931717);  squeeze_dims_71 = None
        mul_tensor_118: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_117, 0.1);  mul_tensor_117 = None
        mul_tensor_119: "f32[384]" = torch.ops.aten.mul.Tensor(arg136_1, 0.9)
        add_tensor_71: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_118, mul_tensor_119);  mul_tensor_118 = mul_tensor_119 = None
        add_tensor_72: "i64[]" = torch.ops.aten.add.Tensor(arg140_1, 1)
        squeeze_dims_72: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_49, [0, 2, 3]);  getitem_49 = None
        squeeze_dims_73: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_24, [0, 2, 3]);  rsqrt_24 = None
        mul_tensor_120: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_72, 0.1)
        mul_tensor_121: "f32[384]" = torch.ops.aten.mul.Tensor(arg141_1, 0.9)
        add_tensor_73: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_120, mul_tensor_121);  mul_tensor_120 = mul_tensor_121 = None
        squeeze_dims_74: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_48, [0, 2, 3]);  getitem_48 = None
        mul_tensor_122: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_74, 1.0006381620931717);  squeeze_dims_74 = None
        mul_tensor_123: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_122, 0.1);  mul_tensor_122 = None
        mul_tensor_124: "f32[384]" = torch.ops.aten.mul.Tensor(arg142_1, 0.9)
        add_tensor_74: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_123, mul_tensor_124);  mul_tensor_123 = mul_tensor_124 = None
        add_tensor_75: "i64[]" = torch.ops.aten.add.Tensor(arg146_1, 1)
        squeeze_dims_75: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_51, [0, 2, 3]);  getitem_51 = None
        squeeze_dims_76: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_25, [0, 2, 3]);  rsqrt_25 = None
        mul_tensor_125: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_75, 0.1)
        mul_tensor_126: "f32[384]" = torch.ops.aten.mul.Tensor(arg147_1, 0.9)
        add_tensor_76: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_125, mul_tensor_126);  mul_tensor_125 = mul_tensor_126 = None
        squeeze_dims_77: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_50, [0, 2, 3]);  getitem_50 = None
        mul_tensor_127: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_77, 1.0006381620931717);  squeeze_dims_77 = None
        mul_tensor_128: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_127, 0.1);  mul_tensor_127 = None
        mul_tensor_129: "f32[384]" = torch.ops.aten.mul.Tensor(arg148_1, 0.9)
        add_tensor_77: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_128, mul_tensor_129);  mul_tensor_128 = mul_tensor_129 = None
        add_tensor_78: "i64[]" = torch.ops.aten.add.Tensor(arg151_1, 1)
        squeeze_dims_78: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_53, [0, 2, 3]);  getitem_53 = None
        squeeze_dims_79: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_26, [0, 2, 3]);  rsqrt_26 = None
        mul_tensor_130: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_78, 0.1)
        mul_tensor_131: "f32[384]" = torch.ops.aten.mul.Tensor(arg152_1, 0.9)
        add_tensor_79: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_130, mul_tensor_131);  mul_tensor_130 = mul_tensor_131 = None
        squeeze_dims_80: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_52, [0, 2, 3]);  getitem_52 = None
        mul_tensor_132: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_80, 1.0006381620931717);  squeeze_dims_80 = None
        mul_tensor_133: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_132, 0.1);  mul_tensor_132 = None
        mul_tensor_134: "f32[384]" = torch.ops.aten.mul.Tensor(arg153_1, 0.9)
        add_tensor_80: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_133, mul_tensor_134);  mul_tensor_133 = mul_tensor_134 = None
        add_tensor_81: "i64[]" = torch.ops.aten.add.Tensor(arg157_1, 1)
        squeeze_dims_81: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_55, [0, 2, 3]);  getitem_55 = None
        squeeze_dims_82: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_27, [0, 2, 3]);  rsqrt_27 = None
        mul_tensor_135: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_81, 0.1)
        mul_tensor_136: "f32[384]" = torch.ops.aten.mul.Tensor(arg158_1, 0.9)
        add_tensor_82: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_135, mul_tensor_136);  mul_tensor_135 = mul_tensor_136 = None
        squeeze_dims_83: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_54, [0, 2, 3]);  getitem_54 = None
        mul_tensor_137: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_83, 1.0006381620931717);  squeeze_dims_83 = None
        mul_tensor_138: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_137, 0.1);  mul_tensor_137 = None
        mul_tensor_139: "f32[384]" = torch.ops.aten.mul.Tensor(arg159_1, 0.9)
        add_tensor_83: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_138, mul_tensor_139);  mul_tensor_138 = mul_tensor_139 = None
        add_tensor_84: "i64[]" = torch.ops.aten.add.Tensor(arg163_1, 1)
        squeeze_dims_84: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_57, [0, 2, 3]);  getitem_57 = None
        squeeze_dims_85: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_28, [0, 2, 3]);  rsqrt_28 = None
        mul_tensor_140: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_84, 0.1)
        mul_tensor_141: "f32[384]" = torch.ops.aten.mul.Tensor(arg164_1, 0.9)
        add_tensor_85: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_140, mul_tensor_141);  mul_tensor_140 = mul_tensor_141 = None
        squeeze_dims_86: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_56, [0, 2, 3]);  getitem_56 = None
        mul_tensor_142: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_86, 1.0006381620931717);  squeeze_dims_86 = None
        mul_tensor_143: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_142, 0.1);  mul_tensor_142 = None
        mul_tensor_144: "f32[384]" = torch.ops.aten.mul.Tensor(arg165_1, 0.9)
        add_tensor_86: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_143, mul_tensor_144);  mul_tensor_143 = mul_tensor_144 = None
        add_tensor_87: "i64[]" = torch.ops.aten.add.Tensor(arg168_1, 1)
        squeeze_dims_87: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_59, [0, 2, 3]);  getitem_59 = None
        squeeze_dims_88: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_29, [0, 2, 3]);  rsqrt_29 = None
        mul_tensor_145: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_87, 0.1)
        mul_tensor_146: "f32[384]" = torch.ops.aten.mul.Tensor(arg169_1, 0.9)
        add_tensor_88: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_145, mul_tensor_146);  mul_tensor_145 = mul_tensor_146 = None
        squeeze_dims_89: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_58, [0, 2, 3]);  getitem_58 = None
        mul_tensor_147: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_89, 1.0006381620931717);  squeeze_dims_89 = None
        mul_tensor_148: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_147, 0.1);  mul_tensor_147 = None
        mul_tensor_149: "f32[384]" = torch.ops.aten.mul.Tensor(arg170_1, 0.9)
        add_tensor_89: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_148, mul_tensor_149);  mul_tensor_148 = mul_tensor_149 = None
        add_tensor_90: "i64[]" = torch.ops.aten.add.Tensor(arg174_1, 1)
        squeeze_dims_90: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_61, [0, 2, 3]);  getitem_61 = None
        squeeze_dims_91: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_30, [0, 2, 3]);  rsqrt_30 = None
        mul_tensor_150: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_90, 0.1)
        mul_tensor_151: "f32[384]" = torch.ops.aten.mul.Tensor(arg175_1, 0.9)
        add_tensor_91: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_150, mul_tensor_151);  mul_tensor_150 = mul_tensor_151 = None
        squeeze_dims_92: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_60, [0, 2, 3]);  getitem_60 = None
        mul_tensor_152: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_92, 1.0006381620931717);  squeeze_dims_92 = None
        mul_tensor_153: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_152, 0.1);  mul_tensor_152 = None
        mul_tensor_154: "f32[384]" = torch.ops.aten.mul.Tensor(arg176_1, 0.9)
        add_tensor_92: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_153, mul_tensor_154);  mul_tensor_153 = mul_tensor_154 = None
        add_tensor_93: "i64[]" = torch.ops.aten.add.Tensor(arg180_1, 1)
        squeeze_dims_93: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_63, [0, 2, 3]);  getitem_63 = None
        squeeze_dims_94: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_31, [0, 2, 3]);  rsqrt_31 = None
        mul_tensor_155: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_93, 0.1)
        mul_tensor_156: "f32[384]" = torch.ops.aten.mul.Tensor(arg181_1, 0.9)
        add_tensor_94: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_155, mul_tensor_156);  mul_tensor_155 = mul_tensor_156 = None
        squeeze_dims_95: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_62, [0, 2, 3]);  getitem_62 = None
        mul_tensor_157: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_95, 1.0006381620931717);  squeeze_dims_95 = None
        mul_tensor_158: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_157, 0.1);  mul_tensor_157 = None
        mul_tensor_159: "f32[384]" = torch.ops.aten.mul.Tensor(arg182_1, 0.9)
        add_tensor_95: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_158, mul_tensor_159);  mul_tensor_158 = mul_tensor_159 = None
        add_tensor_96: "i64[]" = torch.ops.aten.add.Tensor(arg185_1, 1)
        squeeze_dims_96: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_65, [0, 2, 3]);  getitem_65 = None
        squeeze_dims_97: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_32, [0, 2, 3]);  rsqrt_32 = None
        mul_tensor_160: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_96, 0.1)
        mul_tensor_161: "f32[384]" = torch.ops.aten.mul.Tensor(arg186_1, 0.9)
        add_tensor_97: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_160, mul_tensor_161);  mul_tensor_160 = mul_tensor_161 = None
        squeeze_dims_98: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_64, [0, 2, 3]);  getitem_64 = None
        mul_tensor_162: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_98, 1.0006381620931717);  squeeze_dims_98 = None
        mul_tensor_163: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_162, 0.1);  mul_tensor_162 = None
        mul_tensor_164: "f32[384]" = torch.ops.aten.mul.Tensor(arg187_1, 0.9)
        add_tensor_98: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_163, mul_tensor_164);  mul_tensor_163 = mul_tensor_164 = None
        add_tensor_99: "i64[]" = torch.ops.aten.add.Tensor(arg191_1, 1)
        squeeze_dims_99: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_67, [0, 2, 3]);  getitem_67 = None
        squeeze_dims_100: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_33, [0, 2, 3]);  rsqrt_33 = None
        mul_tensor_165: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_99, 0.1)
        mul_tensor_166: "f32[384]" = torch.ops.aten.mul.Tensor(arg192_1, 0.9)
        add_tensor_100: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_165, mul_tensor_166);  mul_tensor_165 = mul_tensor_166 = None
        squeeze_dims_101: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_66, [0, 2, 3]);  getitem_66 = None
        mul_tensor_167: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_101, 1.0006381620931717);  squeeze_dims_101 = None
        mul_tensor_168: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_167, 0.1);  mul_tensor_167 = None
        mul_tensor_169: "f32[384]" = torch.ops.aten.mul.Tensor(arg193_1, 0.9)
        add_tensor_101: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_168, mul_tensor_169);  mul_tensor_168 = mul_tensor_169 = None
        add_tensor_102: "i64[]" = torch.ops.aten.add.Tensor(arg197_1, 1)
        squeeze_dims_102: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_69, [0, 2, 3]);  getitem_69 = None
        squeeze_dims_103: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_34, [0, 2, 3]);  rsqrt_34 = None
        mul_tensor_170: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_102, 0.1)
        mul_tensor_171: "f32[384]" = torch.ops.aten.mul.Tensor(arg198_1, 0.9)
        add_tensor_103: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_170, mul_tensor_171);  mul_tensor_170 = mul_tensor_171 = None
        squeeze_dims_104: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_68, [0, 2, 3]);  getitem_68 = None
        mul_tensor_172: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_104, 1.0006381620931717);  squeeze_dims_104 = None
        mul_tensor_173: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_172, 0.1);  mul_tensor_172 = None
        mul_tensor_174: "f32[384]" = torch.ops.aten.mul.Tensor(arg199_1, 0.9)
        add_tensor_104: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_173, mul_tensor_174);  mul_tensor_173 = mul_tensor_174 = None
        add_tensor_105: "i64[]" = torch.ops.aten.add.Tensor(arg202_1, 1)
        squeeze_dims_105: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_71, [0, 2, 3]);  getitem_71 = None
        squeeze_dims_106: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_35, [0, 2, 3]);  rsqrt_35 = None
        mul_tensor_175: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_105, 0.1)
        mul_tensor_176: "f32[384]" = torch.ops.aten.mul.Tensor(arg203_1, 0.9)
        add_tensor_106: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_175, mul_tensor_176);  mul_tensor_175 = mul_tensor_176 = None
        squeeze_dims_107: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_70, [0, 2, 3]);  getitem_70 = None
        mul_tensor_177: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_107, 1.0006381620931717);  squeeze_dims_107 = None
        mul_tensor_178: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_177, 0.1);  mul_tensor_177 = None
        mul_tensor_179: "f32[384]" = torch.ops.aten.mul.Tensor(arg204_1, 0.9)
        add_tensor_107: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_178, mul_tensor_179);  mul_tensor_178 = mul_tensor_179 = None
        add_tensor_108: "i64[]" = torch.ops.aten.add.Tensor(arg208_1, 1)
        squeeze_dims_108: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_73, [0, 2, 3]);  getitem_73 = None
        squeeze_dims_109: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_36, [0, 2, 3]);  rsqrt_36 = None
        mul_tensor_180: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_108, 0.1)
        mul_tensor_181: "f32[384]" = torch.ops.aten.mul.Tensor(arg209_1, 0.9)
        add_tensor_109: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_180, mul_tensor_181);  mul_tensor_180 = mul_tensor_181 = None
        squeeze_dims_110: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_72, [0, 2, 3]);  getitem_72 = None
        mul_tensor_182: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_110, 1.0006381620931717);  squeeze_dims_110 = None
        mul_tensor_183: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_182, 0.1);  mul_tensor_182 = None
        mul_tensor_184: "f32[384]" = torch.ops.aten.mul.Tensor(arg210_1, 0.9)
        add_tensor_110: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_183, mul_tensor_184);  mul_tensor_183 = mul_tensor_184 = None
        add_tensor_111: "i64[]" = torch.ops.aten.add.Tensor(arg214_1, 1)
        squeeze_dims_111: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_75, [0, 2, 3]);  getitem_75 = None
        squeeze_dims_112: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_37, [0, 2, 3]);  rsqrt_37 = None
        mul_tensor_185: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_111, 0.1)
        mul_tensor_186: "f32[384]" = torch.ops.aten.mul.Tensor(arg215_1, 0.9)
        add_tensor_112: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_185, mul_tensor_186);  mul_tensor_185 = mul_tensor_186 = None
        squeeze_dims_113: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_74, [0, 2, 3]);  getitem_74 = None
        mul_tensor_187: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_113, 1.0006381620931717);  squeeze_dims_113 = None
        mul_tensor_188: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_187, 0.1);  mul_tensor_187 = None
        mul_tensor_189: "f32[384]" = torch.ops.aten.mul.Tensor(arg216_1, 0.9)
        add_tensor_113: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_188, mul_tensor_189);  mul_tensor_188 = mul_tensor_189 = None
        add_tensor_114: "i64[]" = torch.ops.aten.add.Tensor(arg219_1, 1)
        squeeze_dims_114: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_77, [0, 2, 3]);  getitem_77 = None
        squeeze_dims_115: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_38, [0, 2, 3]);  rsqrt_38 = None
        mul_tensor_190: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_114, 0.1)
        mul_tensor_191: "f32[384]" = torch.ops.aten.mul.Tensor(arg220_1, 0.9)
        add_tensor_115: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_190, mul_tensor_191);  mul_tensor_190 = mul_tensor_191 = None
        squeeze_dims_116: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_76, [0, 2, 3]);  getitem_76 = None
        mul_tensor_192: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_116, 1.0006381620931717);  squeeze_dims_116 = None
        mul_tensor_193: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_192, 0.1);  mul_tensor_192 = None
        mul_tensor_194: "f32[384]" = torch.ops.aten.mul.Tensor(arg221_1, 0.9)
        add_tensor_116: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_193, mul_tensor_194);  mul_tensor_193 = mul_tensor_194 = None
        add_tensor_117: "i64[]" = torch.ops.aten.add.Tensor(arg225_1, 1)
        squeeze_dims_117: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_79, [0, 2, 3]);  getitem_79 = None
        squeeze_dims_118: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_39, [0, 2, 3]);  rsqrt_39 = None
        mul_tensor_195: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_117, 0.1)
        mul_tensor_196: "f32[384]" = torch.ops.aten.mul.Tensor(arg226_1, 0.9)
        add_tensor_118: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_195, mul_tensor_196);  mul_tensor_195 = mul_tensor_196 = None
        squeeze_dims_119: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_78, [0, 2, 3]);  getitem_78 = None
        mul_tensor_197: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_119, 1.0006381620931717);  squeeze_dims_119 = None
        mul_tensor_198: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_197, 0.1);  mul_tensor_197 = None
        mul_tensor_199: "f32[384]" = torch.ops.aten.mul.Tensor(arg227_1, 0.9)
        add_tensor_119: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_198, mul_tensor_199);  mul_tensor_198 = mul_tensor_199 = None
        add_tensor_120: "i64[]" = torch.ops.aten.add.Tensor(arg231_1, 1)
        squeeze_dims_120: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_81, [0, 2, 3]);  getitem_81 = None
        squeeze_dims_121: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_40, [0, 2, 3]);  rsqrt_40 = None
        mul_tensor_200: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_120, 0.1)
        mul_tensor_201: "f32[384]" = torch.ops.aten.mul.Tensor(arg232_1, 0.9)
        add_tensor_121: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_200, mul_tensor_201);  mul_tensor_200 = mul_tensor_201 = None
        squeeze_dims_122: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_80, [0, 2, 3]);  getitem_80 = None
        mul_tensor_202: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_122, 1.0006381620931717);  squeeze_dims_122 = None
        mul_tensor_203: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_202, 0.1);  mul_tensor_202 = None
        mul_tensor_204: "f32[384]" = torch.ops.aten.mul.Tensor(arg233_1, 0.9)
        add_tensor_122: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_203, mul_tensor_204);  mul_tensor_203 = mul_tensor_204 = None
        add_tensor_123: "i64[]" = torch.ops.aten.add.Tensor(arg236_1, 1)
        squeeze_dims_123: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_83, [0, 2, 3]);  getitem_83 = None
        squeeze_dims_124: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_41, [0, 2, 3]);  rsqrt_41 = None
        mul_tensor_205: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_123, 0.1)
        mul_tensor_206: "f32[384]" = torch.ops.aten.mul.Tensor(arg237_1, 0.9)
        add_tensor_124: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_205, mul_tensor_206);  mul_tensor_205 = mul_tensor_206 = None
        squeeze_dims_125: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_82, [0, 2, 3]);  getitem_82 = None
        mul_tensor_207: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_125, 1.0006381620931717);  squeeze_dims_125 = None
        mul_tensor_208: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_207, 0.1);  mul_tensor_207 = None
        mul_tensor_209: "f32[384]" = torch.ops.aten.mul.Tensor(arg238_1, 0.9)
        add_tensor_125: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_208, mul_tensor_209);  mul_tensor_208 = mul_tensor_209 = None
        add_tensor_126: "i64[]" = torch.ops.aten.add.Tensor(arg242_1, 1)
        squeeze_dims_126: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_85, [0, 2, 3]);  getitem_85 = None
        squeeze_dims_127: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_42, [0, 2, 3]);  rsqrt_42 = None
        mul_tensor_210: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_126, 0.1)
        mul_tensor_211: "f32[384]" = torch.ops.aten.mul.Tensor(arg243_1, 0.9)
        add_tensor_127: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_210, mul_tensor_211);  mul_tensor_210 = mul_tensor_211 = None
        squeeze_dims_128: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_84, [0, 2, 3]);  getitem_84 = None
        mul_tensor_212: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_128, 1.0006381620931717);  squeeze_dims_128 = None
        mul_tensor_213: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_212, 0.1);  mul_tensor_212 = None
        mul_tensor_214: "f32[384]" = torch.ops.aten.mul.Tensor(arg244_1, 0.9)
        add_tensor_128: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_213, mul_tensor_214);  mul_tensor_213 = mul_tensor_214 = None
        add_tensor_129: "i64[]" = torch.ops.aten.add.Tensor(arg248_1, 1)
        squeeze_dims_129: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_87, [0, 2, 3]);  getitem_87 = None
        squeeze_dims_130: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_43, [0, 2, 3]);  rsqrt_43 = None
        mul_tensor_215: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_129, 0.1)
        mul_tensor_216: "f32[384]" = torch.ops.aten.mul.Tensor(arg249_1, 0.9)
        add_tensor_130: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_215, mul_tensor_216);  mul_tensor_215 = mul_tensor_216 = None
        squeeze_dims_131: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_86, [0, 2, 3]);  getitem_86 = None
        mul_tensor_217: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_131, 1.0006381620931717);  squeeze_dims_131 = None
        mul_tensor_218: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_217, 0.1);  mul_tensor_217 = None
        mul_tensor_219: "f32[384]" = torch.ops.aten.mul.Tensor(arg250_1, 0.9)
        add_tensor_131: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_218, mul_tensor_219);  mul_tensor_218 = mul_tensor_219 = None
        add_tensor_132: "i64[]" = torch.ops.aten.add.Tensor(arg253_1, 1)
        squeeze_dims_132: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_89, [0, 2, 3]);  getitem_89 = None
        squeeze_dims_133: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_44, [0, 2, 3]);  rsqrt_44 = None
        mul_tensor_220: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_132, 0.1)
        mul_tensor_221: "f32[384]" = torch.ops.aten.mul.Tensor(arg254_1, 0.9)
        add_tensor_133: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_220, mul_tensor_221);  mul_tensor_220 = mul_tensor_221 = None
        squeeze_dims_134: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_88, [0, 2, 3]);  getitem_88 = None
        mul_tensor_222: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_134, 1.0006381620931717);  squeeze_dims_134 = None
        mul_tensor_223: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_222, 0.1);  mul_tensor_222 = None
        mul_tensor_224: "f32[384]" = torch.ops.aten.mul.Tensor(arg255_1, 0.9)
        add_tensor_134: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_223, mul_tensor_224);  mul_tensor_223 = mul_tensor_224 = None
        add_tensor_135: "i64[]" = torch.ops.aten.add.Tensor(arg259_1, 1)
        squeeze_dims_135: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_91, [0, 2, 3]);  getitem_91 = None
        squeeze_dims_136: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_45, [0, 2, 3]);  rsqrt_45 = None
        mul_tensor_225: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_135, 0.1)
        mul_tensor_226: "f32[384]" = torch.ops.aten.mul.Tensor(arg260_1, 0.9)
        add_tensor_136: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_225, mul_tensor_226);  mul_tensor_225 = mul_tensor_226 = None
        squeeze_dims_137: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_90, [0, 2, 3]);  getitem_90 = None
        mul_tensor_227: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_137, 1.0006381620931717);  squeeze_dims_137 = None
        mul_tensor_228: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_227, 0.1);  mul_tensor_227 = None
        mul_tensor_229: "f32[384]" = torch.ops.aten.mul.Tensor(arg261_1, 0.9)
        add_tensor_137: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_228, mul_tensor_229);  mul_tensor_228 = mul_tensor_229 = None
        add_tensor_138: "i64[]" = torch.ops.aten.add.Tensor(arg265_1, 1)
        squeeze_dims_138: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_93, [0, 2, 3]);  getitem_93 = None
        squeeze_dims_139: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_46, [0, 2, 3]);  rsqrt_46 = None
        mul_tensor_230: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_138, 0.1)
        mul_tensor_231: "f32[384]" = torch.ops.aten.mul.Tensor(arg266_1, 0.9)
        add_tensor_139: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_230, mul_tensor_231);  mul_tensor_230 = mul_tensor_231 = None
        squeeze_dims_140: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_92, [0, 2, 3]);  getitem_92 = None
        mul_tensor_232: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_140, 1.0006381620931717);  squeeze_dims_140 = None
        mul_tensor_233: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_232, 0.1);  mul_tensor_232 = None
        mul_tensor_234: "f32[384]" = torch.ops.aten.mul.Tensor(arg267_1, 0.9)
        add_tensor_140: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_233, mul_tensor_234);  mul_tensor_233 = mul_tensor_234 = None
        add_tensor_141: "i64[]" = torch.ops.aten.add.Tensor(arg270_1, 1)
        squeeze_dims_141: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_95, [0, 2, 3]);  getitem_95 = None
        squeeze_dims_142: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_47, [0, 2, 3]);  rsqrt_47 = None
        mul_tensor_235: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_141, 0.1)
        mul_tensor_236: "f32[384]" = torch.ops.aten.mul.Tensor(arg271_1, 0.9)
        add_tensor_142: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_235, mul_tensor_236);  mul_tensor_235 = mul_tensor_236 = None
        squeeze_dims_143: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_94, [0, 2, 3]);  getitem_94 = None
        mul_tensor_237: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_143, 1.0006381620931717);  squeeze_dims_143 = None
        mul_tensor_238: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_237, 0.1);  mul_tensor_237 = None
        mul_tensor_239: "f32[384]" = torch.ops.aten.mul.Tensor(arg272_1, 0.9)
        add_tensor_143: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_238, mul_tensor_239);  mul_tensor_238 = mul_tensor_239 = None
        add_tensor_144: "i64[]" = torch.ops.aten.add.Tensor(arg276_1, 1)
        squeeze_dims_144: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_97, [0, 2, 3]);  getitem_97 = None
        squeeze_dims_145: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_48, [0, 2, 3]);  rsqrt_48 = None
        mul_tensor_240: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_144, 0.1)
        mul_tensor_241: "f32[384]" = torch.ops.aten.mul.Tensor(arg277_1, 0.9)
        add_tensor_145: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_240, mul_tensor_241);  mul_tensor_240 = mul_tensor_241 = None
        squeeze_dims_146: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_96, [0, 2, 3]);  getitem_96 = None
        mul_tensor_242: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_146, 1.0006381620931717);  squeeze_dims_146 = None
        mul_tensor_243: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_242, 0.1);  mul_tensor_242 = None
        mul_tensor_244: "f32[384]" = torch.ops.aten.mul.Tensor(arg278_1, 0.9)
        add_tensor_146: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_243, mul_tensor_244);  mul_tensor_243 = mul_tensor_244 = None
        add_tensor_147: "i64[]" = torch.ops.aten.add.Tensor(arg282_1, 1)
        squeeze_dims_147: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_99, [0, 2, 3]);  getitem_99 = None
        squeeze_dims_148: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_49, [0, 2, 3]);  rsqrt_49 = None
        mul_tensor_245: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_147, 0.1)
        mul_tensor_246: "f32[384]" = torch.ops.aten.mul.Tensor(arg283_1, 0.9)
        add_tensor_148: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_245, mul_tensor_246);  mul_tensor_245 = mul_tensor_246 = None
        squeeze_dims_149: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_98, [0, 2, 3]);  getitem_98 = None
        mul_tensor_247: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_149, 1.0006381620931717);  squeeze_dims_149 = None
        mul_tensor_248: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_247, 0.1);  mul_tensor_247 = None
        mul_tensor_249: "f32[384]" = torch.ops.aten.mul.Tensor(arg284_1, 0.9)
        add_tensor_149: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_248, mul_tensor_249);  mul_tensor_248 = mul_tensor_249 = None
        add_tensor_150: "i64[]" = torch.ops.aten.add.Tensor(arg287_1, 1)
        squeeze_dims_150: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_101, [0, 2, 3]);  getitem_101 = None
        squeeze_dims_151: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_50, [0, 2, 3]);  rsqrt_50 = None
        mul_tensor_250: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_150, 0.1)
        mul_tensor_251: "f32[384]" = torch.ops.aten.mul.Tensor(arg288_1, 0.9)
        add_tensor_151: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_250, mul_tensor_251);  mul_tensor_250 = mul_tensor_251 = None
        squeeze_dims_152: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_100, [0, 2, 3]);  getitem_100 = None
        mul_tensor_252: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_152, 1.0006381620931717);  squeeze_dims_152 = None
        mul_tensor_253: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_252, 0.1);  mul_tensor_252 = None
        mul_tensor_254: "f32[384]" = torch.ops.aten.mul.Tensor(arg289_1, 0.9)
        add_tensor_152: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_253, mul_tensor_254);  mul_tensor_253 = mul_tensor_254 = None
        add_tensor_153: "i64[]" = torch.ops.aten.add.Tensor(arg293_1, 1)
        squeeze_dims_153: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_103, [0, 2, 3]);  getitem_103 = None
        squeeze_dims_154: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_51, [0, 2, 3]);  rsqrt_51 = None
        mul_tensor_255: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_153, 0.1)
        mul_tensor_256: "f32[384]" = torch.ops.aten.mul.Tensor(arg294_1, 0.9)
        add_tensor_154: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_255, mul_tensor_256);  mul_tensor_255 = mul_tensor_256 = None
        squeeze_dims_155: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_102, [0, 2, 3]);  getitem_102 = None
        mul_tensor_257: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_155, 1.0006381620931717);  squeeze_dims_155 = None
        mul_tensor_258: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_257, 0.1);  mul_tensor_257 = None
        mul_tensor_259: "f32[384]" = torch.ops.aten.mul.Tensor(arg295_1, 0.9)
        add_tensor_155: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_258, mul_tensor_259);  mul_tensor_258 = mul_tensor_259 = None
        add_tensor_156: "i64[]" = torch.ops.aten.add.Tensor(arg299_1, 1)
        squeeze_dims_156: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_105, [0, 2, 3]);  getitem_105 = None
        squeeze_dims_157: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_52, [0, 2, 3]);  rsqrt_52 = None
        mul_tensor_260: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_156, 0.1)
        mul_tensor_261: "f32[384]" = torch.ops.aten.mul.Tensor(arg300_1, 0.9)
        add_tensor_157: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_260, mul_tensor_261);  mul_tensor_260 = mul_tensor_261 = None
        squeeze_dims_158: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_104, [0, 2, 3]);  getitem_104 = None
        mul_tensor_262: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_158, 1.0006381620931717);  squeeze_dims_158 = None
        mul_tensor_263: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_262, 0.1);  mul_tensor_262 = None
        mul_tensor_264: "f32[384]" = torch.ops.aten.mul.Tensor(arg301_1, 0.9)
        add_tensor_158: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_263, mul_tensor_264);  mul_tensor_263 = mul_tensor_264 = None
        add_tensor_159: "i64[]" = torch.ops.aten.add.Tensor(arg304_1, 1)
        squeeze_dims_159: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_107, [0, 2, 3]);  getitem_107 = None
        squeeze_dims_160: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_53, [0, 2, 3]);  rsqrt_53 = None
        mul_tensor_265: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_159, 0.1)
        mul_tensor_266: "f32[384]" = torch.ops.aten.mul.Tensor(arg305_1, 0.9)
        add_tensor_160: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_265, mul_tensor_266);  mul_tensor_265 = mul_tensor_266 = None
        squeeze_dims_161: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_106, [0, 2, 3]);  getitem_106 = None
        mul_tensor_267: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_161, 1.0006381620931717);  squeeze_dims_161 = None
        mul_tensor_268: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_267, 0.1);  mul_tensor_267 = None
        mul_tensor_269: "f32[384]" = torch.ops.aten.mul.Tensor(arg306_1, 0.9)
        add_tensor_161: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_268, mul_tensor_269);  mul_tensor_268 = mul_tensor_269 = None
        add_tensor_162: "i64[]" = torch.ops.aten.add.Tensor(arg310_1, 1)
        squeeze_dims_162: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_109, [0, 2, 3]);  getitem_109 = None
        squeeze_dims_163: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_54, [0, 2, 3]);  rsqrt_54 = None
        mul_tensor_270: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_162, 0.1)
        mul_tensor_271: "f32[384]" = torch.ops.aten.mul.Tensor(arg311_1, 0.9)
        add_tensor_163: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_270, mul_tensor_271);  mul_tensor_270 = mul_tensor_271 = None
        squeeze_dims_164: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_108, [0, 2, 3]);  getitem_108 = None
        mul_tensor_272: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_164, 1.0006381620931717);  squeeze_dims_164 = None
        mul_tensor_273: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_272, 0.1);  mul_tensor_272 = None
        mul_tensor_274: "f32[384]" = torch.ops.aten.mul.Tensor(arg312_1, 0.9)
        add_tensor_164: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_273, mul_tensor_274);  mul_tensor_273 = mul_tensor_274 = None
        add_tensor_165: "i64[]" = torch.ops.aten.add.Tensor(arg316_1, 1)
        squeeze_dims_165: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_111, [0, 2, 3]);  getitem_111 = None
        squeeze_dims_166: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_55, [0, 2, 3]);  rsqrt_55 = None
        mul_tensor_275: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_165, 0.1)
        mul_tensor_276: "f32[384]" = torch.ops.aten.mul.Tensor(arg317_1, 0.9)
        add_tensor_166: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_275, mul_tensor_276);  mul_tensor_275 = mul_tensor_276 = None
        squeeze_dims_167: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_110, [0, 2, 3]);  getitem_110 = None
        mul_tensor_277: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_167, 1.0006381620931717);  squeeze_dims_167 = None
        mul_tensor_278: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_277, 0.1);  mul_tensor_277 = None
        mul_tensor_279: "f32[384]" = torch.ops.aten.mul.Tensor(arg318_1, 0.9)
        add_tensor_167: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_278, mul_tensor_279);  mul_tensor_278 = mul_tensor_279 = None
        add_tensor_168: "i64[]" = torch.ops.aten.add.Tensor(arg321_1, 1)
        squeeze_dims_168: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_113, [0, 2, 3]);  getitem_113 = None
        squeeze_dims_169: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_56, [0, 2, 3]);  rsqrt_56 = None
        mul_tensor_280: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_168, 0.1)
        mul_tensor_281: "f32[384]" = torch.ops.aten.mul.Tensor(arg322_1, 0.9)
        add_tensor_169: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_280, mul_tensor_281);  mul_tensor_280 = mul_tensor_281 = None
        squeeze_dims_170: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_112, [0, 2, 3]);  getitem_112 = None
        mul_tensor_282: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_170, 1.0006381620931717);  squeeze_dims_170 = None
        mul_tensor_283: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_282, 0.1);  mul_tensor_282 = None
        mul_tensor_284: "f32[384]" = torch.ops.aten.mul.Tensor(arg323_1, 0.9)
        add_tensor_170: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_283, mul_tensor_284);  mul_tensor_283 = mul_tensor_284 = None
        add_tensor_171: "i64[]" = torch.ops.aten.add.Tensor(arg327_1, 1)
        squeeze_dims_171: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_115, [0, 2, 3]);  getitem_115 = None
        squeeze_dims_172: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_57, [0, 2, 3]);  rsqrt_57 = None
        mul_tensor_285: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_171, 0.1)
        mul_tensor_286: "f32[384]" = torch.ops.aten.mul.Tensor(arg328_1, 0.9)
        add_tensor_172: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_285, mul_tensor_286);  mul_tensor_285 = mul_tensor_286 = None
        squeeze_dims_173: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_114, [0, 2, 3]);  getitem_114 = None
        mul_tensor_287: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_173, 1.0006381620931717);  squeeze_dims_173 = None
        mul_tensor_288: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_287, 0.1);  mul_tensor_287 = None
        mul_tensor_289: "f32[384]" = torch.ops.aten.mul.Tensor(arg329_1, 0.9)
        add_tensor_173: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_288, mul_tensor_289);  mul_tensor_288 = mul_tensor_289 = None
        add_tensor_174: "i64[]" = torch.ops.aten.add.Tensor(arg333_1, 1)
        squeeze_dims_174: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_117, [0, 2, 3]);  getitem_117 = None
        squeeze_dims_175: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_58, [0, 2, 3]);  rsqrt_58 = None
        mul_tensor_290: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_174, 0.1)
        mul_tensor_291: "f32[384]" = torch.ops.aten.mul.Tensor(arg334_1, 0.9)
        add_tensor_175: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_290, mul_tensor_291);  mul_tensor_290 = mul_tensor_291 = None
        squeeze_dims_176: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_116, [0, 2, 3]);  getitem_116 = None
        mul_tensor_292: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_176, 1.0006381620931717);  squeeze_dims_176 = None
        mul_tensor_293: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_292, 0.1);  mul_tensor_292 = None
        mul_tensor_294: "f32[384]" = torch.ops.aten.mul.Tensor(arg335_1, 0.9)
        add_tensor_176: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_293, mul_tensor_294);  mul_tensor_293 = mul_tensor_294 = None
        add_tensor_177: "i64[]" = torch.ops.aten.add.Tensor(arg339_1, 1)
        convert_element_type_default: "f32[8, 1408, 7, 7]" = torch.ops.prims.convert_element_type.default(convolution_42, torch.float32)
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_default = None
        getitem_118: "f32[1, 1408, 1, 1]" = var_mean_correction[0]
        getitem_119: "f32[1, 1408, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_178: "f32[1, 1408, 1, 1]" = torch.ops.aten.add.Tensor(getitem_118, 1e-05)
        rsqrt_default: "f32[1, 1408, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor_178);  add_tensor_178 = None
        sub_tensor: "f32[8, 1408, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_42, getitem_119);  convolution_42 = None
        mul_tensor_295: "f32[8, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        squeeze_dims_177: "f32[1408]" = torch.ops.aten.squeeze.dims(getitem_119, [0, 2, 3]);  getitem_119 = None
        mul_tensor_296: "f32[1408]" = torch.ops.aten.mul.Tensor(squeeze_dims_177, 0.1);  squeeze_dims_177 = None
        mul_tensor_297: "f32[1408]" = torch.ops.aten.mul.Tensor(arg340_1, 0.9)
        add_tensor_179: "f32[1408]" = torch.ops.aten.add.Tensor(mul_tensor_296, mul_tensor_297);  mul_tensor_296 = mul_tensor_297 = None
        squeeze_dims_178: "f32[1408]" = torch.ops.aten.squeeze.dims(getitem_118, [0, 2, 3]);  getitem_118 = None
        mul_tensor_298: "f32[1408]" = torch.ops.aten.mul.Tensor(squeeze_dims_178, 1.0025575447570332);  squeeze_dims_178 = None
        mul_tensor_299: "f32[1408]" = torch.ops.aten.mul.Tensor(mul_tensor_298, 0.1);  mul_tensor_298 = None
        mul_tensor_300: "f32[1408]" = torch.ops.aten.mul.Tensor(arg341_1, 0.9)
        add_tensor_180: "f32[1408]" = torch.ops.aten.add.Tensor(mul_tensor_299, mul_tensor_300);  mul_tensor_299 = mul_tensor_300 = None
        unsqueeze_default: "f32[1408, 1]" = torch.ops.aten.unsqueeze.default(arg342_1, -1);  arg342_1 = None
        unsqueeze_default_1: "f32[1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_301: "f32[8, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_295, unsqueeze_default_1);  mul_tensor_295 = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[1408, 1]" = torch.ops.aten.unsqueeze.default(arg343_1, -1);  arg343_1 = None
        unsqueeze_default_3: "f32[1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_181: "f32[8, 1408, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_301, unsqueeze_default_3);  mul_tensor_301 = unsqueeze_default_3 = None
        convert_element_type_default_1: "f16[8, 1408, 7, 7]" = torch.ops.prims.convert_element_type.default(add_tensor_181, torch.float16);  add_tensor_181 = None
        add_tensor_182: "i64[]" = torch.ops.aten.add.Tensor(arg345_1, 1)
        convert_element_type_default_2: "f32[8, 1408, 7, 7]" = torch.ops.prims.convert_element_type.default(convolution_43, torch.float32)
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(convert_element_type_default_2, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_default_2 = None
        getitem_120: "f32[1, 1408, 1, 1]" = var_mean_correction_1[0]
        getitem_121: "f32[1, 1408, 1, 1]" = var_mean_correction_1[1];  var_mean_correction_1 = None
        add_tensor_183: "f32[1, 1408, 1, 1]" = torch.ops.aten.add.Tensor(getitem_120, 1e-05)
        rsqrt_default_1: "f32[1, 1408, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor_183);  add_tensor_183 = None
        sub_tensor_1: "f32[8, 1408, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_43, getitem_121);  convolution_43 = None
        mul_tensor_302: "f32[8, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default_1);  sub_tensor_1 = rsqrt_default_1 = None
        squeeze_dims_179: "f32[1408]" = torch.ops.aten.squeeze.dims(getitem_121, [0, 2, 3]);  getitem_121 = None
        mul_tensor_303: "f32[1408]" = torch.ops.aten.mul.Tensor(squeeze_dims_179, 0.1);  squeeze_dims_179 = None
        mul_tensor_304: "f32[1408]" = torch.ops.aten.mul.Tensor(arg346_1, 0.9)
        add_tensor_184: "f32[1408]" = torch.ops.aten.add.Tensor(mul_tensor_303, mul_tensor_304);  mul_tensor_303 = mul_tensor_304 = None
        squeeze_dims_180: "f32[1408]" = torch.ops.aten.squeeze.dims(getitem_120, [0, 2, 3]);  getitem_120 = None
        mul_tensor_305: "f32[1408]" = torch.ops.aten.mul.Tensor(squeeze_dims_180, 1.0025575447570332);  squeeze_dims_180 = None
        mul_tensor_306: "f32[1408]" = torch.ops.aten.mul.Tensor(mul_tensor_305, 0.1);  mul_tensor_305 = None
        mul_tensor_307: "f32[1408]" = torch.ops.aten.mul.Tensor(arg347_1, 0.9)
        add_tensor_185: "f32[1408]" = torch.ops.aten.add.Tensor(mul_tensor_306, mul_tensor_307);  mul_tensor_306 = mul_tensor_307 = None
        unsqueeze_default_4: "f32[1408, 1]" = torch.ops.aten.unsqueeze.default(arg348_1, -1);  arg348_1 = None
        unsqueeze_default_5: "f32[1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_308: "f32[8, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_302, unsqueeze_default_5);  mul_tensor_302 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[1408, 1]" = torch.ops.aten.unsqueeze.default(arg349_1, -1);  arg349_1 = None
        unsqueeze_default_7: "f32[1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_186: "f32[8, 1408, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_308, unsqueeze_default_7);  mul_tensor_308 = unsqueeze_default_7 = None
        convert_element_type_default_3: "f16[8, 1408, 7, 7]" = torch.ops.prims.convert_element_type.default(add_tensor_186, torch.float16);  add_tensor_186 = None
        add_tensor_187: "f16[8, 1408, 7, 7]" = torch.ops.aten.add.Tensor(convert_element_type_default_1, convert_element_type_default_3);  convert_element_type_default_1 = convert_element_type_default_3 = None
        relu_default: "f16[8, 1408, 7, 7]" = torch.ops.aten.relu.default(add_tensor_187);  add_tensor_187 = None
        mean_dim: "f16[8, 1408, 1, 1]" = torch.ops.aten.mean.dim(relu_default, [-1, -2], True);  relu_default = None
        view_default: "f16[8, 1408]" = torch.ops.aten.view.default(mean_dim, _shape_param_0);  mean_dim = _shape_param_0 = None
        convert_element_type_default_4: "f16[1000]" = torch.ops.prims.convert_element_type.default(arg351_1, torch.float16);  arg351_1 = None
        convert_element_type_default_5: "f16[1000, 1408]" = torch.ops.prims.convert_element_type.default(arg350_1, torch.float16);  arg350_1 = None
        permute_default: "f16[1408, 1000]" = torch.ops.aten.permute.default(convert_element_type_default_5, [1, 0]);  convert_element_type_default_5 = None
        permute_default_1: "f16[1000, 1408]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        unsqueeze_default_8: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_174, 0);  squeeze_dims_174 = None
        unsqueeze_default_9: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, 2);  unsqueeze_default_8 = None
        unsqueeze_default_10: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_9, 3);  unsqueeze_default_9 = None
        unsqueeze_default_11: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_171, 0);  squeeze_dims_171 = None
        unsqueeze_default_12: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, 2);  unsqueeze_default_11 = None
        unsqueeze_default_13: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_12, 3);  unsqueeze_default_12 = None
        unsqueeze_default_14: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_168, 0);  squeeze_dims_168 = None
        unsqueeze_default_15: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, 2);  unsqueeze_default_14 = None
        unsqueeze_default_16: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_15, 3);  unsqueeze_default_15 = None
        unsqueeze_default_17: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_165, 0);  squeeze_dims_165 = None
        unsqueeze_default_18: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_17, 2);  unsqueeze_default_17 = None
        unsqueeze_default_19: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_18, 3);  unsqueeze_default_18 = None
        unsqueeze_default_20: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_162, 0);  squeeze_dims_162 = None
        unsqueeze_default_21: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_20, 2);  unsqueeze_default_20 = None
        unsqueeze_default_22: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_21, 3);  unsqueeze_default_21 = None
        unsqueeze_default_23: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_159, 0);  squeeze_dims_159 = None
        unsqueeze_default_24: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_23, 2);  unsqueeze_default_23 = None
        unsqueeze_default_25: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_24, 3);  unsqueeze_default_24 = None
        unsqueeze_default_26: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_156, 0);  squeeze_dims_156 = None
        unsqueeze_default_27: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_26, 2);  unsqueeze_default_26 = None
        unsqueeze_default_28: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_27, 3);  unsqueeze_default_27 = None
        unsqueeze_default_29: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_153, 0);  squeeze_dims_153 = None
        unsqueeze_default_30: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_29, 2);  unsqueeze_default_29 = None
        unsqueeze_default_31: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_30, 3);  unsqueeze_default_30 = None
        unsqueeze_default_32: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_150, 0);  squeeze_dims_150 = None
        unsqueeze_default_33: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_32, 2);  unsqueeze_default_32 = None
        unsqueeze_default_34: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_33, 3);  unsqueeze_default_33 = None
        unsqueeze_default_35: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_147, 0);  squeeze_dims_147 = None
        unsqueeze_default_36: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_35, 2);  unsqueeze_default_35 = None
        unsqueeze_default_37: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_36, 3);  unsqueeze_default_36 = None
        unsqueeze_default_38: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_144, 0);  squeeze_dims_144 = None
        unsqueeze_default_39: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_38, 2);  unsqueeze_default_38 = None
        unsqueeze_default_40: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_39, 3);  unsqueeze_default_39 = None
        unsqueeze_default_41: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_141, 0);  squeeze_dims_141 = None
        unsqueeze_default_42: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_41, 2);  unsqueeze_default_41 = None
        unsqueeze_default_43: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_42, 3);  unsqueeze_default_42 = None
        unsqueeze_default_44: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_138, 0);  squeeze_dims_138 = None
        unsqueeze_default_45: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_44, 2);  unsqueeze_default_44 = None
        unsqueeze_default_46: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_45, 3);  unsqueeze_default_45 = None
        unsqueeze_default_47: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_135, 0);  squeeze_dims_135 = None
        unsqueeze_default_48: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_47, 2);  unsqueeze_default_47 = None
        unsqueeze_default_49: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_48, 3);  unsqueeze_default_48 = None
        unsqueeze_default_50: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_132, 0);  squeeze_dims_132 = None
        unsqueeze_default_51: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_50, 2);  unsqueeze_default_50 = None
        unsqueeze_default_52: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_51, 3);  unsqueeze_default_51 = None
        unsqueeze_default_53: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_129, 0);  squeeze_dims_129 = None
        unsqueeze_default_54: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_53, 2);  unsqueeze_default_53 = None
        unsqueeze_default_55: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_54, 3);  unsqueeze_default_54 = None
        unsqueeze_default_56: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_126, 0);  squeeze_dims_126 = None
        unsqueeze_default_57: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_56, 2);  unsqueeze_default_56 = None
        unsqueeze_default_58: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_57, 3);  unsqueeze_default_57 = None
        unsqueeze_default_59: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_123, 0);  squeeze_dims_123 = None
        unsqueeze_default_60: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_59, 2);  unsqueeze_default_59 = None
        unsqueeze_default_61: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_60, 3);  unsqueeze_default_60 = None
        unsqueeze_default_62: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_120, 0);  squeeze_dims_120 = None
        unsqueeze_default_63: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_62, 2);  unsqueeze_default_62 = None
        unsqueeze_default_64: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_63, 3);  unsqueeze_default_63 = None
        unsqueeze_default_65: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_117, 0);  squeeze_dims_117 = None
        unsqueeze_default_66: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_65, 2);  unsqueeze_default_65 = None
        unsqueeze_default_67: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_66, 3);  unsqueeze_default_66 = None
        unsqueeze_default_68: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_114, 0);  squeeze_dims_114 = None
        unsqueeze_default_69: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_68, 2);  unsqueeze_default_68 = None
        unsqueeze_default_70: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_69, 3);  unsqueeze_default_69 = None
        unsqueeze_default_71: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_111, 0);  squeeze_dims_111 = None
        unsqueeze_default_72: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_71, 2);  unsqueeze_default_71 = None
        unsqueeze_default_73: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_72, 3);  unsqueeze_default_72 = None
        unsqueeze_default_74: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_108, 0);  squeeze_dims_108 = None
        unsqueeze_default_75: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_74, 2);  unsqueeze_default_74 = None
        unsqueeze_default_76: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_75, 3);  unsqueeze_default_75 = None
        unsqueeze_default_77: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_105, 0);  squeeze_dims_105 = None
        unsqueeze_default_78: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_77, 2);  unsqueeze_default_77 = None
        unsqueeze_default_79: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_78, 3);  unsqueeze_default_78 = None
        unsqueeze_default_80: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_102, 0);  squeeze_dims_102 = None
        unsqueeze_default_81: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_80, 2);  unsqueeze_default_80 = None
        unsqueeze_default_82: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_81, 3);  unsqueeze_default_81 = None
        unsqueeze_default_83: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_99, 0);  squeeze_dims_99 = None
        unsqueeze_default_84: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_83, 2);  unsqueeze_default_83 = None
        unsqueeze_default_85: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_84, 3);  unsqueeze_default_84 = None
        unsqueeze_default_86: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_96, 0);  squeeze_dims_96 = None
        unsqueeze_default_87: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_86, 2);  unsqueeze_default_86 = None
        unsqueeze_default_88: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_87, 3);  unsqueeze_default_87 = None
        unsqueeze_default_89: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_93, 0);  squeeze_dims_93 = None
        unsqueeze_default_90: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_89, 2);  unsqueeze_default_89 = None
        unsqueeze_default_91: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_90, 3);  unsqueeze_default_90 = None
        unsqueeze_default_92: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_90, 0);  squeeze_dims_90 = None
        unsqueeze_default_93: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_92, 2);  unsqueeze_default_92 = None
        unsqueeze_default_94: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_93, 3);  unsqueeze_default_93 = None
        unsqueeze_default_95: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_87, 0);  squeeze_dims_87 = None
        unsqueeze_default_96: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_95, 2);  unsqueeze_default_95 = None
        unsqueeze_default_97: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_96, 3);  unsqueeze_default_96 = None
        unsqueeze_default_98: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_84, 0);  squeeze_dims_84 = None
        unsqueeze_default_99: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_98, 2);  unsqueeze_default_98 = None
        unsqueeze_default_100: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_99, 3);  unsqueeze_default_99 = None
        unsqueeze_default_101: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_81, 0);  squeeze_dims_81 = None
        unsqueeze_default_102: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_101, 2);  unsqueeze_default_101 = None
        unsqueeze_default_103: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_102, 3);  unsqueeze_default_102 = None
        unsqueeze_default_104: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_78, 0);  squeeze_dims_78 = None
        unsqueeze_default_105: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_104, 2);  unsqueeze_default_104 = None
        unsqueeze_default_106: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_105, 3);  unsqueeze_default_105 = None
        unsqueeze_default_107: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_75, 0);  squeeze_dims_75 = None
        unsqueeze_default_108: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_107, 2);  unsqueeze_default_107 = None
        unsqueeze_default_109: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_108, 3);  unsqueeze_default_108 = None
        unsqueeze_default_110: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_72, 0);  squeeze_dims_72 = None
        unsqueeze_default_111: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_110, 2);  unsqueeze_default_110 = None
        unsqueeze_default_112: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_111, 3);  unsqueeze_default_111 = None
        unsqueeze_default_113: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_69, 0);  squeeze_dims_69 = None
        unsqueeze_default_114: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_113, 2);  unsqueeze_default_113 = None
        unsqueeze_default_115: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_114, 3);  unsqueeze_default_114 = None
        unsqueeze_default_116: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_66, 0);  squeeze_dims_66 = None
        unsqueeze_default_117: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_116, 2);  unsqueeze_default_116 = None
        unsqueeze_default_118: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_117, 3);  unsqueeze_default_117 = None
        unsqueeze_default_119: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_63, 0);  squeeze_dims_63 = None
        unsqueeze_default_120: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_119, 2);  unsqueeze_default_119 = None
        unsqueeze_default_121: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_120, 3);  unsqueeze_default_120 = None
        unsqueeze_default_122: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_60, 0);  squeeze_dims_60 = None
        unsqueeze_default_123: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_122, 2);  unsqueeze_default_122 = None
        unsqueeze_default_124: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_123, 3);  unsqueeze_default_123 = None
        unsqueeze_default_125: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_57, 0);  squeeze_dims_57 = None
        unsqueeze_default_126: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_125, 2);  unsqueeze_default_125 = None
        unsqueeze_default_127: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_126, 3);  unsqueeze_default_126 = None
        unsqueeze_default_128: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_54, 0);  squeeze_dims_54 = None
        unsqueeze_default_129: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_128, 2);  unsqueeze_default_128 = None
        unsqueeze_default_130: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_129, 3);  unsqueeze_default_129 = None
        unsqueeze_default_131: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_dims_51, 0);  squeeze_dims_51 = None
        unsqueeze_default_132: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_131, 2);  unsqueeze_default_131 = None
        unsqueeze_default_133: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_132, 3);  unsqueeze_default_132 = None
        unsqueeze_default_134: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_dims_48, 0);  squeeze_dims_48 = None
        unsqueeze_default_135: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_134, 2);  unsqueeze_default_134 = None
        unsqueeze_default_136: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_135, 3);  unsqueeze_default_135 = None
        unsqueeze_default_137: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_dims_45, 0);  squeeze_dims_45 = None
        unsqueeze_default_138: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_137, 2);  unsqueeze_default_137 = None
        unsqueeze_default_139: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_138, 3);  unsqueeze_default_138 = None
        unsqueeze_default_140: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_dims_42, 0);  squeeze_dims_42 = None
        unsqueeze_default_141: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_140, 2);  unsqueeze_default_140 = None
        unsqueeze_default_142: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_141, 3);  unsqueeze_default_141 = None
        unsqueeze_default_143: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_dims_39, 0);  squeeze_dims_39 = None
        unsqueeze_default_144: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_143, 2);  unsqueeze_default_143 = None
        unsqueeze_default_145: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_144, 3);  unsqueeze_default_144 = None
        unsqueeze_default_146: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_dims_36, 0);  squeeze_dims_36 = None
        unsqueeze_default_147: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_146, 2);  unsqueeze_default_146 = None
        unsqueeze_default_148: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_147, 3);  unsqueeze_default_147 = None
        unsqueeze_default_149: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_dims_33, 0);  squeeze_dims_33 = None
        unsqueeze_default_150: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_149, 2);  unsqueeze_default_149 = None
        unsqueeze_default_151: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_150, 3);  unsqueeze_default_150 = None
        unsqueeze_default_152: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_dims_30, 0);  squeeze_dims_30 = None
        unsqueeze_default_153: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_152, 2);  unsqueeze_default_152 = None
        unsqueeze_default_154: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_153, 3);  unsqueeze_default_153 = None
        unsqueeze_default_155: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_dims_27, 0);  squeeze_dims_27 = None
        unsqueeze_default_156: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_155, 2);  unsqueeze_default_155 = None
        unsqueeze_default_157: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_156, 3);  unsqueeze_default_156 = None
        unsqueeze_default_158: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_dims_24, 0);  squeeze_dims_24 = None
        unsqueeze_default_159: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_158, 2);  unsqueeze_default_158 = None
        unsqueeze_default_160: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_159, 3);  unsqueeze_default_159 = None
        unsqueeze_default_161: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_dims_21, 0);  squeeze_dims_21 = None
        unsqueeze_default_162: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_161, 2);  unsqueeze_default_161 = None
        unsqueeze_default_163: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_162, 3);  unsqueeze_default_162 = None
        unsqueeze_default_164: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(squeeze_dims_18, 0);  squeeze_dims_18 = None
        unsqueeze_default_165: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_164, 2);  unsqueeze_default_164 = None
        unsqueeze_default_166: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_165, 3);  unsqueeze_default_165 = None
        unsqueeze_default_167: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(squeeze_dims_15, 0);  squeeze_dims_15 = None
        unsqueeze_default_168: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_167, 2);  unsqueeze_default_167 = None
        unsqueeze_default_169: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_168, 3);  unsqueeze_default_168 = None
        unsqueeze_default_170: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(squeeze_dims_12, 0);  squeeze_dims_12 = None
        unsqueeze_default_171: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_170, 2);  unsqueeze_default_170 = None
        unsqueeze_default_172: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_171, 3);  unsqueeze_default_171 = None
        unsqueeze_default_173: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(squeeze_dims_9, 0);  squeeze_dims_9 = None
        unsqueeze_default_174: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_173, 2);  unsqueeze_default_173 = None
        unsqueeze_default_175: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_174, 3);  unsqueeze_default_174 = None
        unsqueeze_default_176: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(squeeze_dims_6, 0);  squeeze_dims_6 = None
        unsqueeze_default_177: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_176, 2);  unsqueeze_default_176 = None
        unsqueeze_default_178: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_177, 3);  unsqueeze_default_177 = None
        unsqueeze_default_179: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_dims_3, 0);  squeeze_dims_3 = None
        unsqueeze_default_180: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_179, 2);  unsqueeze_default_179 = None
        unsqueeze_default_181: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_180, 3);  unsqueeze_default_180 = None
        unsqueeze_default_182: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_183: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_182, 2);  unsqueeze_default_182 = None
        unsqueeze_default_184: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_183, 3);  unsqueeze_default_183 = None
        copy__default: "i64[]" = torch.ops.aten.copy_.default(arg2_1, add_tensor);  arg2_1 = add_tensor = None
        copy__default_1: "f32[64]" = torch.ops.aten.copy_.default(arg3_1, add_tensor_1);  arg3_1 = add_tensor_1 = None
        copy__default_2: "f32[64]" = torch.ops.aten.copy_.default(arg4_1, add_tensor_2);  arg4_1 = add_tensor_2 = None
        copy__default_3: "i64[]" = torch.ops.aten.copy_.default(arg8_1, add_tensor_3);  arg8_1 = add_tensor_3 = None
        copy__default_4: "f32[64]" = torch.ops.aten.copy_.default(arg9_1, add_tensor_4);  arg9_1 = add_tensor_4 = None
        copy__default_5: "f32[64]" = torch.ops.aten.copy_.default(arg10_1, add_tensor_5);  arg10_1 = add_tensor_5 = None
        copy__default_6: "i64[]" = torch.ops.aten.copy_.default(arg14_1, add_tensor_6);  arg14_1 = add_tensor_6 = None
        copy__default_7: "f32[96]" = torch.ops.aten.copy_.default(arg15_1, add_tensor_7);  arg15_1 = add_tensor_7 = None
        copy__default_8: "f32[96]" = torch.ops.aten.copy_.default(arg16_1, add_tensor_8);  arg16_1 = add_tensor_8 = None
        copy__default_9: "i64[]" = torch.ops.aten.copy_.default(arg20_1, add_tensor_9);  arg20_1 = add_tensor_9 = None
        copy__default_10: "f32[96]" = torch.ops.aten.copy_.default(arg21_1, add_tensor_10);  arg21_1 = add_tensor_10 = None
        copy__default_11: "f32[96]" = torch.ops.aten.copy_.default(arg22_1, add_tensor_11);  arg22_1 = add_tensor_11 = None
        copy__default_12: "i64[]" = torch.ops.aten.copy_.default(arg25_1, add_tensor_12);  arg25_1 = add_tensor_12 = None
        copy__default_13: "f32[96]" = torch.ops.aten.copy_.default(arg26_1, add_tensor_13);  arg26_1 = add_tensor_13 = None
        copy__default_14: "f32[96]" = torch.ops.aten.copy_.default(arg27_1, add_tensor_14);  arg27_1 = add_tensor_14 = None
        copy__default_15: "i64[]" = torch.ops.aten.copy_.default(arg31_1, add_tensor_15);  arg31_1 = add_tensor_15 = None
        copy__default_16: "f32[96]" = torch.ops.aten.copy_.default(arg32_1, add_tensor_16);  arg32_1 = add_tensor_16 = None
        copy__default_17: "f32[96]" = torch.ops.aten.copy_.default(arg33_1, add_tensor_17);  arg33_1 = add_tensor_17 = None
        copy__default_18: "i64[]" = torch.ops.aten.copy_.default(arg37_1, add_tensor_18);  arg37_1 = add_tensor_18 = None
        copy__default_19: "f32[96]" = torch.ops.aten.copy_.default(arg38_1, add_tensor_19);  arg38_1 = add_tensor_19 = None
        copy__default_20: "f32[96]" = torch.ops.aten.copy_.default(arg39_1, add_tensor_20);  arg39_1 = add_tensor_20 = None
        copy__default_21: "i64[]" = torch.ops.aten.copy_.default(arg43_1, add_tensor_21);  arg43_1 = add_tensor_21 = None
        copy__default_22: "f32[192]" = torch.ops.aten.copy_.default(arg44_1, add_tensor_22);  arg44_1 = add_tensor_22 = None
        copy__default_23: "f32[192]" = torch.ops.aten.copy_.default(arg45_1, add_tensor_23);  arg45_1 = add_tensor_23 = None
        copy__default_24: "i64[]" = torch.ops.aten.copy_.default(arg49_1, add_tensor_24);  arg49_1 = add_tensor_24 = None
        copy__default_25: "f32[192]" = torch.ops.aten.copy_.default(arg50_1, add_tensor_25);  arg50_1 = add_tensor_25 = None
        copy__default_26: "f32[192]" = torch.ops.aten.copy_.default(arg51_1, add_tensor_26);  arg51_1 = add_tensor_26 = None
        copy__default_27: "i64[]" = torch.ops.aten.copy_.default(arg54_1, add_tensor_27);  arg54_1 = add_tensor_27 = None
        copy__default_28: "f32[192]" = torch.ops.aten.copy_.default(arg55_1, add_tensor_28);  arg55_1 = add_tensor_28 = None
        copy__default_29: "f32[192]" = torch.ops.aten.copy_.default(arg56_1, add_tensor_29);  arg56_1 = add_tensor_29 = None
        copy__default_30: "i64[]" = torch.ops.aten.copy_.default(arg60_1, add_tensor_30);  arg60_1 = add_tensor_30 = None
        copy__default_31: "f32[192]" = torch.ops.aten.copy_.default(arg61_1, add_tensor_31);  arg61_1 = add_tensor_31 = None
        copy__default_32: "f32[192]" = torch.ops.aten.copy_.default(arg62_1, add_tensor_32);  arg62_1 = add_tensor_32 = None
        copy__default_33: "i64[]" = torch.ops.aten.copy_.default(arg66_1, add_tensor_33);  arg66_1 = add_tensor_33 = None
        copy__default_34: "f32[192]" = torch.ops.aten.copy_.default(arg67_1, add_tensor_34);  arg67_1 = add_tensor_34 = None
        copy__default_35: "f32[192]" = torch.ops.aten.copy_.default(arg68_1, add_tensor_35);  arg68_1 = add_tensor_35 = None
        copy__default_36: "i64[]" = torch.ops.aten.copy_.default(arg71_1, add_tensor_36);  arg71_1 = add_tensor_36 = None
        copy__default_37: "f32[192]" = torch.ops.aten.copy_.default(arg72_1, add_tensor_37);  arg72_1 = add_tensor_37 = None
        copy__default_38: "f32[192]" = torch.ops.aten.copy_.default(arg73_1, add_tensor_38);  arg73_1 = add_tensor_38 = None
        copy__default_39: "i64[]" = torch.ops.aten.copy_.default(arg77_1, add_tensor_39);  arg77_1 = add_tensor_39 = None
        copy__default_40: "f32[192]" = torch.ops.aten.copy_.default(arg78_1, add_tensor_40);  arg78_1 = add_tensor_40 = None
        copy__default_41: "f32[192]" = torch.ops.aten.copy_.default(arg79_1, add_tensor_41);  arg79_1 = add_tensor_41 = None
        copy__default_42: "i64[]" = torch.ops.aten.copy_.default(arg83_1, add_tensor_42);  arg83_1 = add_tensor_42 = None
        copy__default_43: "f32[192]" = torch.ops.aten.copy_.default(arg84_1, add_tensor_43);  arg84_1 = add_tensor_43 = None
        copy__default_44: "f32[192]" = torch.ops.aten.copy_.default(arg85_1, add_tensor_44);  arg85_1 = add_tensor_44 = None
        copy__default_45: "i64[]" = torch.ops.aten.copy_.default(arg88_1, add_tensor_45);  arg88_1 = add_tensor_45 = None
        copy__default_46: "f32[192]" = torch.ops.aten.copy_.default(arg89_1, add_tensor_46);  arg89_1 = add_tensor_46 = None
        copy__default_47: "f32[192]" = torch.ops.aten.copy_.default(arg90_1, add_tensor_47);  arg90_1 = add_tensor_47 = None
        copy__default_48: "i64[]" = torch.ops.aten.copy_.default(arg94_1, add_tensor_48);  arg94_1 = add_tensor_48 = None
        copy__default_49: "f32[192]" = torch.ops.aten.copy_.default(arg95_1, add_tensor_49);  arg95_1 = add_tensor_49 = None
        copy__default_50: "f32[192]" = torch.ops.aten.copy_.default(arg96_1, add_tensor_50);  arg96_1 = add_tensor_50 = None
        copy__default_51: "i64[]" = torch.ops.aten.copy_.default(arg100_1, add_tensor_51);  arg100_1 = add_tensor_51 = None
        copy__default_52: "f32[192]" = torch.ops.aten.copy_.default(arg101_1, add_tensor_52);  arg101_1 = add_tensor_52 = None
        copy__default_53: "f32[192]" = torch.ops.aten.copy_.default(arg102_1, add_tensor_53);  arg102_1 = add_tensor_53 = None
        copy__default_54: "i64[]" = torch.ops.aten.copy_.default(arg106_1, add_tensor_54);  arg106_1 = add_tensor_54 = None
        copy__default_55: "f32[384]" = torch.ops.aten.copy_.default(arg107_1, add_tensor_55);  arg107_1 = add_tensor_55 = None
        copy__default_56: "f32[384]" = torch.ops.aten.copy_.default(arg108_1, add_tensor_56);  arg108_1 = add_tensor_56 = None
        copy__default_57: "i64[]" = torch.ops.aten.copy_.default(arg112_1, add_tensor_57);  arg112_1 = add_tensor_57 = None
        copy__default_58: "f32[384]" = torch.ops.aten.copy_.default(arg113_1, add_tensor_58);  arg113_1 = add_tensor_58 = None
        copy__default_59: "f32[384]" = torch.ops.aten.copy_.default(arg114_1, add_tensor_59);  arg114_1 = add_tensor_59 = None
        copy__default_60: "i64[]" = torch.ops.aten.copy_.default(arg117_1, add_tensor_60);  arg117_1 = add_tensor_60 = None
        copy__default_61: "f32[384]" = torch.ops.aten.copy_.default(arg118_1, add_tensor_61);  arg118_1 = add_tensor_61 = None
        copy__default_62: "f32[384]" = torch.ops.aten.copy_.default(arg119_1, add_tensor_62);  arg119_1 = add_tensor_62 = None
        copy__default_63: "i64[]" = torch.ops.aten.copy_.default(arg123_1, add_tensor_63);  arg123_1 = add_tensor_63 = None
        copy__default_64: "f32[384]" = torch.ops.aten.copy_.default(arg124_1, add_tensor_64);  arg124_1 = add_tensor_64 = None
        copy__default_65: "f32[384]" = torch.ops.aten.copy_.default(arg125_1, add_tensor_65);  arg125_1 = add_tensor_65 = None
        copy__default_66: "i64[]" = torch.ops.aten.copy_.default(arg129_1, add_tensor_66);  arg129_1 = add_tensor_66 = None
        copy__default_67: "f32[384]" = torch.ops.aten.copy_.default(arg130_1, add_tensor_67);  arg130_1 = add_tensor_67 = None
        copy__default_68: "f32[384]" = torch.ops.aten.copy_.default(arg131_1, add_tensor_68);  arg131_1 = add_tensor_68 = None
        copy__default_69: "i64[]" = torch.ops.aten.copy_.default(arg134_1, add_tensor_69);  arg134_1 = add_tensor_69 = None
        copy__default_70: "f32[384]" = torch.ops.aten.copy_.default(arg135_1, add_tensor_70);  arg135_1 = add_tensor_70 = None
        copy__default_71: "f32[384]" = torch.ops.aten.copy_.default(arg136_1, add_tensor_71);  arg136_1 = add_tensor_71 = None
        copy__default_72: "i64[]" = torch.ops.aten.copy_.default(arg140_1, add_tensor_72);  arg140_1 = add_tensor_72 = None
        copy__default_73: "f32[384]" = torch.ops.aten.copy_.default(arg141_1, add_tensor_73);  arg141_1 = add_tensor_73 = None
        copy__default_74: "f32[384]" = torch.ops.aten.copy_.default(arg142_1, add_tensor_74);  arg142_1 = add_tensor_74 = None
        copy__default_75: "i64[]" = torch.ops.aten.copy_.default(arg146_1, add_tensor_75);  arg146_1 = add_tensor_75 = None
        copy__default_76: "f32[384]" = torch.ops.aten.copy_.default(arg147_1, add_tensor_76);  arg147_1 = add_tensor_76 = None
        copy__default_77: "f32[384]" = torch.ops.aten.copy_.default(arg148_1, add_tensor_77);  arg148_1 = add_tensor_77 = None
        copy__default_78: "i64[]" = torch.ops.aten.copy_.default(arg151_1, add_tensor_78);  arg151_1 = add_tensor_78 = None
        copy__default_79: "f32[384]" = torch.ops.aten.copy_.default(arg152_1, add_tensor_79);  arg152_1 = add_tensor_79 = None
        copy__default_80: "f32[384]" = torch.ops.aten.copy_.default(arg153_1, add_tensor_80);  arg153_1 = add_tensor_80 = None
        copy__default_81: "i64[]" = torch.ops.aten.copy_.default(arg157_1, add_tensor_81);  arg157_1 = add_tensor_81 = None
        copy__default_82: "f32[384]" = torch.ops.aten.copy_.default(arg158_1, add_tensor_82);  arg158_1 = add_tensor_82 = None
        copy__default_83: "f32[384]" = torch.ops.aten.copy_.default(arg159_1, add_tensor_83);  arg159_1 = add_tensor_83 = None
        copy__default_84: "i64[]" = torch.ops.aten.copy_.default(arg163_1, add_tensor_84);  arg163_1 = add_tensor_84 = None
        copy__default_85: "f32[384]" = torch.ops.aten.copy_.default(arg164_1, add_tensor_85);  arg164_1 = add_tensor_85 = None
        copy__default_86: "f32[384]" = torch.ops.aten.copy_.default(arg165_1, add_tensor_86);  arg165_1 = add_tensor_86 = None
        copy__default_87: "i64[]" = torch.ops.aten.copy_.default(arg168_1, add_tensor_87);  arg168_1 = add_tensor_87 = None
        copy__default_88: "f32[384]" = torch.ops.aten.copy_.default(arg169_1, add_tensor_88);  arg169_1 = add_tensor_88 = None
        copy__default_89: "f32[384]" = torch.ops.aten.copy_.default(arg170_1, add_tensor_89);  arg170_1 = add_tensor_89 = None
        copy__default_90: "i64[]" = torch.ops.aten.copy_.default(arg174_1, add_tensor_90);  arg174_1 = add_tensor_90 = None
        copy__default_91: "f32[384]" = torch.ops.aten.copy_.default(arg175_1, add_tensor_91);  arg175_1 = add_tensor_91 = None
        copy__default_92: "f32[384]" = torch.ops.aten.copy_.default(arg176_1, add_tensor_92);  arg176_1 = add_tensor_92 = None
        copy__default_93: "i64[]" = torch.ops.aten.copy_.default(arg180_1, add_tensor_93);  arg180_1 = add_tensor_93 = None
        copy__default_94: "f32[384]" = torch.ops.aten.copy_.default(arg181_1, add_tensor_94);  arg181_1 = add_tensor_94 = None
        copy__default_95: "f32[384]" = torch.ops.aten.copy_.default(arg182_1, add_tensor_95);  arg182_1 = add_tensor_95 = None
        copy__default_96: "i64[]" = torch.ops.aten.copy_.default(arg185_1, add_tensor_96);  arg185_1 = add_tensor_96 = None
        copy__default_97: "f32[384]" = torch.ops.aten.copy_.default(arg186_1, add_tensor_97);  arg186_1 = add_tensor_97 = None
        copy__default_98: "f32[384]" = torch.ops.aten.copy_.default(arg187_1, add_tensor_98);  arg187_1 = add_tensor_98 = None
        copy__default_99: "i64[]" = torch.ops.aten.copy_.default(arg191_1, add_tensor_99);  arg191_1 = add_tensor_99 = None
        copy__default_100: "f32[384]" = torch.ops.aten.copy_.default(arg192_1, add_tensor_100);  arg192_1 = add_tensor_100 = None
        copy__default_101: "f32[384]" = torch.ops.aten.copy_.default(arg193_1, add_tensor_101);  arg193_1 = add_tensor_101 = None
        copy__default_102: "i64[]" = torch.ops.aten.copy_.default(arg197_1, add_tensor_102);  arg197_1 = add_tensor_102 = None
        copy__default_103: "f32[384]" = torch.ops.aten.copy_.default(arg198_1, add_tensor_103);  arg198_1 = add_tensor_103 = None
        copy__default_104: "f32[384]" = torch.ops.aten.copy_.default(arg199_1, add_tensor_104);  arg199_1 = add_tensor_104 = None
        copy__default_105: "i64[]" = torch.ops.aten.copy_.default(arg202_1, add_tensor_105);  arg202_1 = add_tensor_105 = None
        copy__default_106: "f32[384]" = torch.ops.aten.copy_.default(arg203_1, add_tensor_106);  arg203_1 = add_tensor_106 = None
        copy__default_107: "f32[384]" = torch.ops.aten.copy_.default(arg204_1, add_tensor_107);  arg204_1 = add_tensor_107 = None
        copy__default_108: "i64[]" = torch.ops.aten.copy_.default(arg208_1, add_tensor_108);  arg208_1 = add_tensor_108 = None
        copy__default_109: "f32[384]" = torch.ops.aten.copy_.default(arg209_1, add_tensor_109);  arg209_1 = add_tensor_109 = None
        copy__default_110: "f32[384]" = torch.ops.aten.copy_.default(arg210_1, add_tensor_110);  arg210_1 = add_tensor_110 = None
        copy__default_111: "i64[]" = torch.ops.aten.copy_.default(arg214_1, add_tensor_111);  arg214_1 = add_tensor_111 = None
        copy__default_112: "f32[384]" = torch.ops.aten.copy_.default(arg215_1, add_tensor_112);  arg215_1 = add_tensor_112 = None
        copy__default_113: "f32[384]" = torch.ops.aten.copy_.default(arg216_1, add_tensor_113);  arg216_1 = add_tensor_113 = None
        copy__default_114: "i64[]" = torch.ops.aten.copy_.default(arg219_1, add_tensor_114);  arg219_1 = add_tensor_114 = None
        copy__default_115: "f32[384]" = torch.ops.aten.copy_.default(arg220_1, add_tensor_115);  arg220_1 = add_tensor_115 = None
        copy__default_116: "f32[384]" = torch.ops.aten.copy_.default(arg221_1, add_tensor_116);  arg221_1 = add_tensor_116 = None
        copy__default_117: "i64[]" = torch.ops.aten.copy_.default(arg225_1, add_tensor_117);  arg225_1 = add_tensor_117 = None
        copy__default_118: "f32[384]" = torch.ops.aten.copy_.default(arg226_1, add_tensor_118);  arg226_1 = add_tensor_118 = None
        copy__default_119: "f32[384]" = torch.ops.aten.copy_.default(arg227_1, add_tensor_119);  arg227_1 = add_tensor_119 = None
        copy__default_120: "i64[]" = torch.ops.aten.copy_.default(arg231_1, add_tensor_120);  arg231_1 = add_tensor_120 = None
        copy__default_121: "f32[384]" = torch.ops.aten.copy_.default(arg232_1, add_tensor_121);  arg232_1 = add_tensor_121 = None
        copy__default_122: "f32[384]" = torch.ops.aten.copy_.default(arg233_1, add_tensor_122);  arg233_1 = add_tensor_122 = None
        copy__default_123: "i64[]" = torch.ops.aten.copy_.default(arg236_1, add_tensor_123);  arg236_1 = add_tensor_123 = None
        copy__default_124: "f32[384]" = torch.ops.aten.copy_.default(arg237_1, add_tensor_124);  arg237_1 = add_tensor_124 = None
        copy__default_125: "f32[384]" = torch.ops.aten.copy_.default(arg238_1, add_tensor_125);  arg238_1 = add_tensor_125 = None
        copy__default_126: "i64[]" = torch.ops.aten.copy_.default(arg242_1, add_tensor_126);  arg242_1 = add_tensor_126 = None
        copy__default_127: "f32[384]" = torch.ops.aten.copy_.default(arg243_1, add_tensor_127);  arg243_1 = add_tensor_127 = None
        copy__default_128: "f32[384]" = torch.ops.aten.copy_.default(arg244_1, add_tensor_128);  arg244_1 = add_tensor_128 = None
        copy__default_129: "i64[]" = torch.ops.aten.copy_.default(arg248_1, add_tensor_129);  arg248_1 = add_tensor_129 = None
        copy__default_130: "f32[384]" = torch.ops.aten.copy_.default(arg249_1, add_tensor_130);  arg249_1 = add_tensor_130 = None
        copy__default_131: "f32[384]" = torch.ops.aten.copy_.default(arg250_1, add_tensor_131);  arg250_1 = add_tensor_131 = None
        copy__default_132: "i64[]" = torch.ops.aten.copy_.default(arg253_1, add_tensor_132);  arg253_1 = add_tensor_132 = None
        copy__default_133: "f32[384]" = torch.ops.aten.copy_.default(arg254_1, add_tensor_133);  arg254_1 = add_tensor_133 = None
        copy__default_134: "f32[384]" = torch.ops.aten.copy_.default(arg255_1, add_tensor_134);  arg255_1 = add_tensor_134 = None
        copy__default_135: "i64[]" = torch.ops.aten.copy_.default(arg259_1, add_tensor_135);  arg259_1 = add_tensor_135 = None
        copy__default_136: "f32[384]" = torch.ops.aten.copy_.default(arg260_1, add_tensor_136);  arg260_1 = add_tensor_136 = None
        copy__default_137: "f32[384]" = torch.ops.aten.copy_.default(arg261_1, add_tensor_137);  arg261_1 = add_tensor_137 = None
        copy__default_138: "i64[]" = torch.ops.aten.copy_.default(arg265_1, add_tensor_138);  arg265_1 = add_tensor_138 = None
        copy__default_139: "f32[384]" = torch.ops.aten.copy_.default(arg266_1, add_tensor_139);  arg266_1 = add_tensor_139 = None
        copy__default_140: "f32[384]" = torch.ops.aten.copy_.default(arg267_1, add_tensor_140);  arg267_1 = add_tensor_140 = None
        copy__default_141: "i64[]" = torch.ops.aten.copy_.default(arg270_1, add_tensor_141);  arg270_1 = add_tensor_141 = None
        copy__default_142: "f32[384]" = torch.ops.aten.copy_.default(arg271_1, add_tensor_142);  arg271_1 = add_tensor_142 = None
        copy__default_143: "f32[384]" = torch.ops.aten.copy_.default(arg272_1, add_tensor_143);  arg272_1 = add_tensor_143 = None
        copy__default_144: "i64[]" = torch.ops.aten.copy_.default(arg276_1, add_tensor_144);  arg276_1 = add_tensor_144 = None
        copy__default_145: "f32[384]" = torch.ops.aten.copy_.default(arg277_1, add_tensor_145);  arg277_1 = add_tensor_145 = None
        copy__default_146: "f32[384]" = torch.ops.aten.copy_.default(arg278_1, add_tensor_146);  arg278_1 = add_tensor_146 = None
        copy__default_147: "i64[]" = torch.ops.aten.copy_.default(arg282_1, add_tensor_147);  arg282_1 = add_tensor_147 = None
        copy__default_148: "f32[384]" = torch.ops.aten.copy_.default(arg283_1, add_tensor_148);  arg283_1 = add_tensor_148 = None
        copy__default_149: "f32[384]" = torch.ops.aten.copy_.default(arg284_1, add_tensor_149);  arg284_1 = add_tensor_149 = None
        copy__default_150: "i64[]" = torch.ops.aten.copy_.default(arg287_1, add_tensor_150);  arg287_1 = add_tensor_150 = None
        copy__default_151: "f32[384]" = torch.ops.aten.copy_.default(arg288_1, add_tensor_151);  arg288_1 = add_tensor_151 = None
        copy__default_152: "f32[384]" = torch.ops.aten.copy_.default(arg289_1, add_tensor_152);  arg289_1 = add_tensor_152 = None
        copy__default_153: "i64[]" = torch.ops.aten.copy_.default(arg293_1, add_tensor_153);  arg293_1 = add_tensor_153 = None
        copy__default_154: "f32[384]" = torch.ops.aten.copy_.default(arg294_1, add_tensor_154);  arg294_1 = add_tensor_154 = None
        copy__default_155: "f32[384]" = torch.ops.aten.copy_.default(arg295_1, add_tensor_155);  arg295_1 = add_tensor_155 = None
        copy__default_156: "i64[]" = torch.ops.aten.copy_.default(arg299_1, add_tensor_156);  arg299_1 = add_tensor_156 = None
        copy__default_157: "f32[384]" = torch.ops.aten.copy_.default(arg300_1, add_tensor_157);  arg300_1 = add_tensor_157 = None
        copy__default_158: "f32[384]" = torch.ops.aten.copy_.default(arg301_1, add_tensor_158);  arg301_1 = add_tensor_158 = None
        copy__default_159: "i64[]" = torch.ops.aten.copy_.default(arg304_1, add_tensor_159);  arg304_1 = add_tensor_159 = None
        copy__default_160: "f32[384]" = torch.ops.aten.copy_.default(arg305_1, add_tensor_160);  arg305_1 = add_tensor_160 = None
        copy__default_161: "f32[384]" = torch.ops.aten.copy_.default(arg306_1, add_tensor_161);  arg306_1 = add_tensor_161 = None
        copy__default_162: "i64[]" = torch.ops.aten.copy_.default(arg310_1, add_tensor_162);  arg310_1 = add_tensor_162 = None
        copy__default_163: "f32[384]" = torch.ops.aten.copy_.default(arg311_1, add_tensor_163);  arg311_1 = add_tensor_163 = None
        copy__default_164: "f32[384]" = torch.ops.aten.copy_.default(arg312_1, add_tensor_164);  arg312_1 = add_tensor_164 = None
        copy__default_165: "i64[]" = torch.ops.aten.copy_.default(arg316_1, add_tensor_165);  arg316_1 = add_tensor_165 = None
        copy__default_166: "f32[384]" = torch.ops.aten.copy_.default(arg317_1, add_tensor_166);  arg317_1 = add_tensor_166 = None
        copy__default_167: "f32[384]" = torch.ops.aten.copy_.default(arg318_1, add_tensor_167);  arg318_1 = add_tensor_167 = None
        copy__default_168: "i64[]" = torch.ops.aten.copy_.default(arg321_1, add_tensor_168);  arg321_1 = add_tensor_168 = None
        copy__default_169: "f32[384]" = torch.ops.aten.copy_.default(arg322_1, add_tensor_169);  arg322_1 = add_tensor_169 = None
        copy__default_170: "f32[384]" = torch.ops.aten.copy_.default(arg323_1, add_tensor_170);  arg323_1 = add_tensor_170 = None
        copy__default_171: "i64[]" = torch.ops.aten.copy_.default(arg327_1, add_tensor_171);  arg327_1 = add_tensor_171 = None
        copy__default_172: "f32[384]" = torch.ops.aten.copy_.default(arg328_1, add_tensor_172);  arg328_1 = add_tensor_172 = None
        copy__default_173: "f32[384]" = torch.ops.aten.copy_.default(arg329_1, add_tensor_173);  arg329_1 = add_tensor_173 = None
        copy__default_174: "i64[]" = torch.ops.aten.copy_.default(arg333_1, add_tensor_174);  arg333_1 = add_tensor_174 = None
        copy__default_175: "f32[384]" = torch.ops.aten.copy_.default(arg334_1, add_tensor_175);  arg334_1 = add_tensor_175 = None
        copy__default_176: "f32[384]" = torch.ops.aten.copy_.default(arg335_1, add_tensor_176);  arg335_1 = add_tensor_176 = None
        copy__default_177: "i64[]" = torch.ops.aten.copy_.default(arg339_1, add_tensor_177);  arg339_1 = add_tensor_177 = None
        copy__default_178: "f32[1408]" = torch.ops.aten.copy_.default(arg340_1, add_tensor_179);  arg340_1 = add_tensor_179 = None
        copy__default_179: "f32[1408]" = torch.ops.aten.copy_.default(arg341_1, add_tensor_180);  arg341_1 = add_tensor_180 = None
        copy__default_180: "i64[]" = torch.ops.aten.copy_.default(arg345_1, add_tensor_182);  arg345_1 = add_tensor_182 = None
        copy__default_181: "f32[1408]" = torch.ops.aten.copy_.default(arg346_1, add_tensor_184);  arg346_1 = add_tensor_184 = None
        copy__default_182: "f32[1408]" = torch.ops.aten.copy_.default(arg347_1, add_tensor_185);  arg347_1 = add_tensor_185 = None
        return (squeeze_dims_1, squeeze_dims_4, squeeze_dims_7, squeeze_dims_10, squeeze_dims_13, squeeze_dims_16, squeeze_dims_19, squeeze_dims_22, squeeze_dims_25, squeeze_dims_28, squeeze_dims_31, squeeze_dims_34, squeeze_dims_37, squeeze_dims_40, squeeze_dims_43, squeeze_dims_46, squeeze_dims_49, squeeze_dims_52, squeeze_dims_55, squeeze_dims_58, squeeze_dims_61, squeeze_dims_64, squeeze_dims_67, squeeze_dims_70, squeeze_dims_73, squeeze_dims_76, squeeze_dims_79, squeeze_dims_82, squeeze_dims_85, squeeze_dims_88, squeeze_dims_91, squeeze_dims_94, squeeze_dims_97, squeeze_dims_100, squeeze_dims_103, squeeze_dims_106, squeeze_dims_109, squeeze_dims_112, squeeze_dims_115, squeeze_dims_118, squeeze_dims_121, squeeze_dims_124, squeeze_dims_127, squeeze_dims_130, squeeze_dims_133, squeeze_dims_136, squeeze_dims_139, squeeze_dims_142, squeeze_dims_145, squeeze_dims_148, squeeze_dims_151, squeeze_dims_154, squeeze_dims_157, squeeze_dims_160, squeeze_dims_163, squeeze_dims_166, squeeze_dims_169, squeeze_dims_172, squeeze_dims_175, view_default, convert_element_type_default_4, permute_default_1, unsqueeze_default_10, unsqueeze_default_13, unsqueeze_default_16, unsqueeze_default_19, unsqueeze_default_22, unsqueeze_default_25, unsqueeze_default_28, unsqueeze_default_31, unsqueeze_default_34, unsqueeze_default_37, unsqueeze_default_40, unsqueeze_default_43, unsqueeze_default_46, unsqueeze_default_49, unsqueeze_default_52, unsqueeze_default_55, unsqueeze_default_58, unsqueeze_default_61, unsqueeze_default_64, unsqueeze_default_67, unsqueeze_default_70, unsqueeze_default_73, unsqueeze_default_76, unsqueeze_default_79, unsqueeze_default_82, unsqueeze_default_85, unsqueeze_default_88, unsqueeze_default_91, unsqueeze_default_94, unsqueeze_default_97, unsqueeze_default_100, unsqueeze_default_103, unsqueeze_default_106, unsqueeze_default_109, unsqueeze_default_112, unsqueeze_default_115, unsqueeze_default_118, unsqueeze_default_121, unsqueeze_default_124, unsqueeze_default_127, unsqueeze_default_130, unsqueeze_default_133, unsqueeze_default_136, unsqueeze_default_139, unsqueeze_default_142, unsqueeze_default_145, unsqueeze_default_148, unsqueeze_default_151, unsqueeze_default_154, unsqueeze_default_157, unsqueeze_default_160, unsqueeze_default_163, unsqueeze_default_166, unsqueeze_default_169, unsqueeze_default_172, unsqueeze_default_175, unsqueeze_default_178, unsqueeze_default_181, unsqueeze_default_184, copy__default, copy__default_1, copy__default_2, copy__default_3, copy__default_4, copy__default_5, copy__default_6, copy__default_7, copy__default_8, copy__default_9, copy__default_10, copy__default_11, copy__default_12, copy__default_13, copy__default_14, copy__default_15, copy__default_16, copy__default_17, copy__default_18, copy__default_19, copy__default_20, copy__default_21, copy__default_22, copy__default_23, copy__default_24, copy__default_25, copy__default_26, copy__default_27, copy__default_28, copy__default_29, copy__default_30, copy__default_31, copy__default_32, copy__default_33, copy__default_34, copy__default_35, copy__default_36, copy__default_37, copy__default_38, copy__default_39, copy__default_40, copy__default_41, copy__default_42, copy__default_43, copy__default_44, copy__default_45, copy__default_46, copy__default_47, copy__default_48, copy__default_49, copy__default_50, copy__default_51, copy__default_52, copy__default_53, copy__default_54, copy__default_55, copy__default_56, copy__default_57, copy__default_58, copy__default_59, copy__default_60, copy__default_61, copy__default_62, copy__default_63, copy__default_64, copy__default_65, copy__default_66, copy__default_67, copy__default_68, copy__default_69, copy__default_70, copy__default_71, copy__default_72, copy__default_73, copy__default_74, copy__default_75, copy__default_76, copy__default_77, copy__default_78, copy__default_79, copy__default_80, copy__default_81, copy__default_82, copy__default_83, copy__default_84, copy__default_85, copy__default_86, copy__default_87, copy__default_88, copy__default_89, copy__default_90, copy__default_91, copy__default_92, copy__default_93, copy__default_94, copy__default_95, copy__default_96, copy__default_97, copy__default_98, copy__default_99, copy__default_100, copy__default_101, copy__default_102, copy__default_103, copy__default_104, copy__default_105, copy__default_106, copy__default_107, copy__default_108, copy__default_109, copy__default_110, copy__default_111, copy__default_112, copy__default_113, copy__default_114, copy__default_115, copy__default_116, copy__default_117, copy__default_118, copy__default_119, copy__default_120, copy__default_121, copy__default_122, copy__default_123, copy__default_124, copy__default_125, copy__default_126, copy__default_127, copy__default_128, copy__default_129, copy__default_130, copy__default_131, copy__default_132, copy__default_133, copy__default_134, copy__default_135, copy__default_136, copy__default_137, copy__default_138, copy__default_139, copy__default_140, copy__default_141, copy__default_142, copy__default_143, copy__default_144, copy__default_145, copy__default_146, copy__default_147, copy__default_148, copy__default_149, copy__default_150, copy__default_151, copy__default_152, copy__default_153, copy__default_154, copy__default_155, copy__default_156, copy__default_157, copy__default_158, copy__default_159, copy__default_160, copy__default_161, copy__default_162, copy__default_163, copy__default_164, copy__default_165, copy__default_166, copy__default_167, copy__default_168, copy__default_169, copy__default_170, copy__default_171, copy__default_172, copy__default_173, copy__default_174, copy__default_175, copy__default_176, copy__default_177, copy__default_178, copy__default_179, copy__default_180, copy__default_181, copy__default_182)


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
