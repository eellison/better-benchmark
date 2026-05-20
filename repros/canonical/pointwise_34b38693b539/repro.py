"""
Standalone repro captured via capture_hook.
Label: inductor_timm_perf_cuda_h100-4-7-linux.aws.h100_graph53
Pattern hash: 34b38693b539
Shape hash: 233db019
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([], i64, max=2), T([1, 16, 1, 1], f32), T([1, 16, 1, 1], f32), T([16], f32), T([1, 16, 1, 1], f32), T([16], f32), T([], i64, max=2), T([1, 8, 1, 1], f32), T([1, 8, 1, 1], f32), T([8], f32), T([1, 8, 1, 1], f32), T([8], f32), T([], i64, max=2), T([1, 8, 1, 1], f32), T([8], f32), T([1, 8, 1, 1], f32), T([8], f32), T([], i64, max=2), T([1, 8, 1, 1], f32), T([1, 8, 1, 1], f32), T([8], f32), T([1, 8, 1, 1], f32), T([8], f32), T([], i64, max=2), T([1, 8, 1, 1], f32), T([1, 8, 1, 1], f32), T([8], f32), T([1, 8, 1, 1], f32), T([8], f32), T([], i64, max=2), T([1, 24, 1, 1], f32), T([1, 24, 1, 1], f32), T([24], f32), T([1, 24, 1, 1], f32), T([24], f32), T([], i64, max=2), T([1, 24, 1, 1], f32), T([24], f32), T([1, 24, 1, 1], f32), T([24], f32), T([], i64, max=2), T([1, 48, 1, 1], f32), T([1, 48, 1, 1], f32), T([48], f32), T([1, 48, 1, 1], f32), T([48], f32), T([], i64, max=2), T([1, 12, 1, 1], f32), T([1, 12, 1, 1], f32), T([12], f32), T([1, 12, 1, 1], f32), T([12], f32), T([], i64, max=2), T([1, 12, 1, 1], f32), T([1, 12, 1, 1], f32), T([12], f32), T([1, 12, 1, 1], f32), T([12], f32), T([], i64, max=2), T([1, 16, 1, 1], f32), T([1, 16, 1, 1], f32), T([16], f32), T([1, 16, 1, 1], f32), T([16], f32), T([], i64, max=2), T([1, 24, 1, 1], f32), T([1, 24, 1, 1], f32), T([24], f32), T([1, 24, 1, 1], f32), T([24], f32), T([], i64, max=2), T([1, 36, 1, 1], f32), T([1, 36, 1, 1], f32), T([36], f32), T([1, 36, 1, 1], f32), T([36], f32), T([], i64, max=2), T([1, 36, 1, 1], f32), T([36], f32), T([1, 36, 1, 1], f32), T([36], f32), T([], i64, max=2), T([1, 12, 1, 1], f32), T([1, 12, 1, 1], f32), T([12], f32), T([1, 12, 1, 1], f32), T([12], f32), T([], i64, max=2), T([1, 12, 1, 1], f32), T([1, 12, 1, 1], f32), T([12], f32), T([1, 12, 1, 1], f32), T([12], f32), T([], i64, max=2), T([1, 36, 1, 1], f32), T([1, 36, 1, 1], f32), T([36], f32), T([1, 36, 1, 1], f32), T([36], f32), T([], i64, max=2), T([1, 36, 1, 1], f32), T([36], f32), T([1, 36, 1, 1], f32), T([36], f32), T([], i64, max=2), T([1, 72, 1, 1], f32), T([72], f32), T([1, 72, 1, 1], f32), T([72], f32), T([], i64, max=2), T([1, 20, 1, 1], f32), T([1, 20, 1, 1], f32), T([20], f32), T([1, 20, 1, 1], f32), T([20], f32), T([], i64, max=2), T([1, 20, 1, 1], f32), T([1, 20, 1, 1], f32), T([20], f32), T([1, 20, 1, 1], f32), T([20], f32), T([], i64, max=2), T([1, 24, 1, 1], f32), T([1, 24, 1, 1], f32), T([24], f32), T([1, 24, 1, 1], f32), T([24], f32), T([], i64, max=2), T([1, 40, 1, 1], f32), T([1, 40, 1, 1], f32), T([40], f32), T([1, 40, 1, 1], f32), T([40], f32), T([], i64, max=2), T([1, 60, 1, 1], f32), T([1, 60, 1, 1], f32), T([60], f32), T([1, 60, 1, 1], f32), T([60], f32), T([], i64, max=2), T([1, 60, 1, 1], f32), T([60], f32), T([1, 60, 1, 1], f32), T([60], f32), T([], i64, max=2), T([1, 20, 1, 1], f32), T([1, 20, 1, 1], f32), T([20], f32), T([1, 20, 1, 1], f32), T([20], f32), T([], i64, max=2), T([1, 20, 1, 1], f32), T([1, 20, 1, 1], f32), T([20], f32), T([1, 20, 1, 1], f32), T([20], f32), T([], i64, max=2), T([1, 120, 1, 1], f32), T([1, 120, 1, 1], f32), T([120], f32), T([1, 120, 1, 1], f32), T([120], f32), T([], i64, max=2), T([1, 120, 1, 1], f32), T([120], f32), T([1, 120, 1, 1], f32), T([120], f32), T([], i64, max=2), T([1, 240, 1, 1], f32), T([1, 240, 1, 1], f32), T([240], f32), T([1, 240, 1, 1], f32), T([240], f32), T([], i64, max=2), T([1, 40, 1, 1], f32), T([1, 40, 1, 1], f32), T([40], f32), T([1, 40, 1, 1], f32), T([40], f32), T([], i64, max=2), T([1, 40, 1, 1], f32), T([1, 40, 1, 1], f32), T([40], f32), T([1, 40, 1, 1], f32), T([40], f32), T([], i64, max=2), T([1, 40, 1, 1], f32), T([1, 40, 1, 1], f32), T([40], f32), T([1, 40, 1, 1], f32), T([40], f32), T([], i64, max=2), T([1, 80, 1, 1], f32), T([1, 80, 1, 1], f32), T([80], f32), T([1, 80, 1, 1], f32), T([80], f32), T([], i64, max=2), T([1, 100, 1, 1], f32), T([1, 100, 1, 1], f32), T([100], f32), T([1, 100, 1, 1], f32), T([100], f32), T([], i64, max=2), T([1, 100, 1, 1], f32), T([100], f32), T([1, 100, 1, 1], f32), T([100], f32), T([], i64, max=2), T([1, 40, 1, 1], f32), T([1, 40, 1, 1], f32), T([40], f32), T([1, 40, 1, 1], f32), T([40], f32), T([], i64, max=2), T([1, 40, 1, 1], f32), T([1, 40, 1, 1], f32), T([40], f32), T([1, 40, 1, 1], f32), T([40], f32), T([], i64, max=2), T([1, 92, 1, 1], f32), T([1, 92, 1, 1], f32), T([92], f32), T([1, 92, 1, 1], f32), T([92], f32), T([], i64, max=2), T([1, 92, 1, 1], f32), T([92], f32), T([1, 92, 1, 1], f32), T([92], f32), T([], i64, max=2), T([1, 40, 1, 1], f32), T([1, 40, 1, 1], f32), T([40], f32), T([1, 40, 1, 1], f32), T([40], f32), T([], i64, max=2), T([1, 40, 1, 1], f32), T([1, 40, 1, 1], f32), T([40], f32), T([1, 40, 1, 1], f32), T([40], f32), T([], i64, max=2), T([1, 92, 1, 1], f32), T([1, 92, 1, 1], f32), T([92], f32), T([1, 92, 1, 1], f32), T([92], f32), T([], i64, max=2), T([1, 92, 1, 1], f32), T([92], f32), T([1, 92, 1, 1], f32), T([92], f32), T([], i64, max=2), T([1, 40, 1, 1], f32), T([1, 40, 1, 1], f32), T([40], f32), T([1, 40, 1, 1], f32), T([40], f32), T([], i64, max=2), T([1, 40, 1, 1], f32), T([1, 40, 1, 1], f32), T([40], f32), T([1, 40, 1, 1], f32), T([40], f32), T([], i64, max=2), T([1, 240, 1, 1], f32), T([1, 240, 1, 1], f32), T([240], f32), T([1, 240, 1, 1], f32), T([240], f32), T([], i64, max=2), T([1, 240, 1, 1], f32), T([240], f32), T([1, 240, 1, 1], f32), T([240], f32), T([], i64, max=2), T([1, 56, 1, 1], f32), T([1, 56, 1, 1], f32), T([56], f32), T([1, 56, 1, 1], f32), T([56], f32), T([], i64, max=2), T([1, 56, 1, 1], f32), T([1, 56, 1, 1], f32), T([56], f32), T([1, 56, 1, 1], f32), T([56], f32), T([], i64, max=2), T([1, 80, 1, 1], f32), T([1, 80, 1, 1], f32), T([80], f32), T([1, 80, 1, 1], f32), T([80], f32), T([], i64, max=2), T([1, 112, 1, 1], f32), T([1, 112, 1, 1], f32), T([112], f32), T([1, 112, 1, 1], f32), T([112], f32), T([], i64, max=2), T([1, 336, 1, 1], f32), T([1, 336, 1, 1], f32), T([336], f32), T([1, 336, 1, 1], f32), T([336], f32), T([], i64, max=2), T([1, 336, 1, 1], f32), T([336], f32), T([1, 336, 1, 1], f32), T([336], f32), T([], i64, max=2), T([1, 56, 1, 1], f32), T([1, 56, 1, 1], f32), T([56], f32), T([1, 56, 1, 1], f32), T([56], f32), T([], i64, max=2), T([1, 56, 1, 1], f32), T([1, 56, 1, 1], f32), T([56], f32), T([1, 56, 1, 1], f32), T([56], f32), T([], i64, max=2), T([1, 336, 1, 1], f32), T([1, 336, 1, 1], f32), T([336], f32), T([1, 336, 1, 1], f32), T([336], f32), T([], i64, max=2), T([1, 336, 1, 1], f32), T([336], f32), T([1, 336, 1, 1], f32), T([336], f32), T([], i64, max=2), T([1, 672, 1, 1], f32), T([672], f32), T([1, 672, 1, 1], f32), T([672], f32), T([], i64, max=2), T([1, 80, 1, 1], f32), T([1, 80, 1, 1], f32), T([80], f32), T([1, 80, 1, 1], f32), T([80], f32), T([], i64, max=2), T([1, 80, 1, 1], f32), T([1, 80, 1, 1], f32), T([80], f32), T([1, 80, 1, 1], f32), T([80], f32), T([], i64, max=2), T([1, 112, 1, 1], f32), T([1, 112, 1, 1], f32), T([112], f32), T([1, 112, 1, 1], f32), T([112], f32), T([], i64, max=2), T([1, 160, 1, 1], f32), T([1, 160, 1, 1], f32), T([160], f32), T([1, 160, 1, 1], f32), T([160], f32), T([], i64, max=2), T([1, 480, 1, 1], f32), T([1, 480, 1, 1], f32), T([480], f32), T([1, 480, 1, 1], f32), T([480], f32), T([], i64, max=2), T([1, 480, 1, 1], f32), T([480], f32), T([1, 480, 1, 1], f32), T([480], f32), T([], i64, max=2), T([1, 80, 1, 1], f32), T([1, 80, 1, 1], f32), T([80], f32), T([1, 80, 1, 1], f32), T([80], f32), T([], i64, max=2), T([1, 80, 1, 1], f32), T([1, 80, 1, 1], f32), T([80], f32), T([1, 80, 1, 1], f32), T([80], f32), T([], i64, max=2), T([1, 480, 1, 1], f32), T([1, 480, 1, 1], f32), T([480], f32), T([1, 480, 1, 1], f32), T([480], f32), T([], i64, max=2), T([1, 480, 1, 1], f32), T([480], f32), T([1, 480, 1, 1], f32), T([480], f32), T([], i64, max=2), T([1, 80, 1, 1], f32), T([1, 80, 1, 1], f32), T([80], f32), T([1, 80, 1, 1], f32), T([80], f32), T([], i64, max=2), T([1, 80, 1, 1], f32), T([1, 80, 1, 1], f32), T([80], f32), T([1, 80, 1, 1], f32), T([80], f32), T([], i64, max=2), T([1, 480, 1, 1], f32), T([1, 480, 1, 1], f32), T([480], f32), T([1, 480, 1, 1], f32), T([480], f32), T([], i64, max=2), T([1, 480, 1, 1], f32), T([480], f32), T([1, 480, 1, 1], f32), T([480], f32), T([], i64, max=2), T([1, 80, 1, 1], f32), T([1, 80, 1, 1], f32), T([80], f32), T([1, 80, 1, 1], f32), T([80], f32), T([], i64, max=2), T([1, 80, 1, 1], f32), T([1, 80, 1, 1], f32), T([80], f32), T([1, 80, 1, 1], f32), T([80], f32), T([], i64, max=2), T([1, 480, 1, 1], f32), T([1, 480, 1, 1], f32), T([480], f32), T([1, 480, 1, 1], f32), T([480], f32), T([], i64, max=2), T([1, 480, 1, 1], f32), T([480], f32), T([1, 480, 1, 1], f32), T([480], f32), T([], i64, max=2), T([1, 80, 1, 1], f32), T([1, 80, 1, 1], f32), T([80], f32), T([1, 80, 1, 1], f32), T([80], f32), T([], i64, max=2), T([1, 80, 1, 1], f32), T([1, 80, 1, 1], f32), T([80], f32), T([1, 80, 1, 1], f32), T([80], f32), T([], i64, max=2), T([1, 960, 1, 1], f32), T([960], f32), T([1, 960, 1, 1], f32), T([960], f32), T([8, 1280, 1, 1], f16), T([1000], f32), T([1000, 1280], f32), S([8, 1280]))"

class Repro(torch.nn.Module):
    def forward(self, arg2_1: "i64[]", getitem_1: "f32[1, 16, 1, 1]", rsqrt: "f32[1, 16, 1, 1]", arg3_1: "f32[16]", getitem: "f32[1, 16, 1, 1]", arg4_1: "f32[16]", arg8_1: "i64[]", getitem_3: "f32[1, 8, 1, 1]", rsqrt_1: "f32[1, 8, 1, 1]", arg9_1: "f32[8]", getitem_2: "f32[1, 8, 1, 1]", arg10_1: "f32[8]", arg14_1: "i64[]", getitem_5: "f32[1, 8, 1, 1]", arg15_1: "f32[8]", getitem_4: "f32[1, 8, 1, 1]", arg16_1: "f32[8]", arg20_1: "i64[]", getitem_7: "f32[1, 8, 1, 1]", rsqrt_3: "f32[1, 8, 1, 1]", arg21_1: "f32[8]", getitem_6: "f32[1, 8, 1, 1]", arg22_1: "f32[8]", arg26_1: "i64[]", getitem_9: "f32[1, 8, 1, 1]", rsqrt_4: "f32[1, 8, 1, 1]", arg27_1: "f32[8]", getitem_8: "f32[1, 8, 1, 1]", arg28_1: "f32[8]", arg32_1: "i64[]", getitem_11: "f32[1, 24, 1, 1]", rsqrt_5: "f32[1, 24, 1, 1]", arg33_1: "f32[24]", getitem_10: "f32[1, 24, 1, 1]", arg34_1: "f32[24]", arg38_1: "i64[]", getitem_13: "f32[1, 24, 1, 1]", arg39_1: "f32[24]", getitem_12: "f32[1, 24, 1, 1]", arg40_1: "f32[24]", arg44_1: "i64[]", getitem_15: "f32[1, 48, 1, 1]", rsqrt_7: "f32[1, 48, 1, 1]", arg45_1: "f32[48]", getitem_14: "f32[1, 48, 1, 1]", arg46_1: "f32[48]", arg50_1: "i64[]", getitem_17: "f32[1, 12, 1, 1]", rsqrt_8: "f32[1, 12, 1, 1]", arg51_1: "f32[12]", getitem_16: "f32[1, 12, 1, 1]", arg52_1: "f32[12]", arg56_1: "i64[]", getitem_19: "f32[1, 12, 1, 1]", rsqrt_9: "f32[1, 12, 1, 1]", arg57_1: "f32[12]", getitem_18: "f32[1, 12, 1, 1]", arg58_1: "f32[12]", arg62_1: "i64[]", getitem_21: "f32[1, 16, 1, 1]", rsqrt_10: "f32[1, 16, 1, 1]", arg63_1: "f32[16]", getitem_20: "f32[1, 16, 1, 1]", arg64_1: "f32[16]", arg68_1: "i64[]", getitem_23: "f32[1, 24, 1, 1]", rsqrt_11: "f32[1, 24, 1, 1]", arg69_1: "f32[24]", getitem_22: "f32[1, 24, 1, 1]", arg70_1: "f32[24]", arg74_1: "i64[]", getitem_25: "f32[1, 36, 1, 1]", rsqrt_12: "f32[1, 36, 1, 1]", arg75_1: "f32[36]", getitem_24: "f32[1, 36, 1, 1]", arg76_1: "f32[36]", arg80_1: "i64[]", getitem_27: "f32[1, 36, 1, 1]", arg81_1: "f32[36]", getitem_26: "f32[1, 36, 1, 1]", arg82_1: "f32[36]", arg86_1: "i64[]", getitem_29: "f32[1, 12, 1, 1]", rsqrt_14: "f32[1, 12, 1, 1]", arg87_1: "f32[12]", getitem_28: "f32[1, 12, 1, 1]", arg88_1: "f32[12]", arg92_1: "i64[]", getitem_31: "f32[1, 12, 1, 1]", rsqrt_15: "f32[1, 12, 1, 1]", arg93_1: "f32[12]", getitem_30: "f32[1, 12, 1, 1]", arg94_1: "f32[12]", arg98_1: "i64[]", getitem_33: "f32[1, 36, 1, 1]", rsqrt_16: "f32[1, 36, 1, 1]", arg99_1: "f32[36]", getitem_32: "f32[1, 36, 1, 1]", arg100_1: "f32[36]", arg104_1: "i64[]", getitem_35: "f32[1, 36, 1, 1]", arg105_1: "f32[36]", getitem_34: "f32[1, 36, 1, 1]", arg106_1: "f32[36]", arg110_1: "i64[]", getitem_37: "f32[1, 72, 1, 1]", arg111_1: "f32[72]", getitem_36: "f32[1, 72, 1, 1]", arg112_1: "f32[72]", arg120_1: "i64[]", getitem_39: "f32[1, 20, 1, 1]", rsqrt_19: "f32[1, 20, 1, 1]", arg121_1: "f32[20]", getitem_38: "f32[1, 20, 1, 1]", arg122_1: "f32[20]", arg126_1: "i64[]", getitem_41: "f32[1, 20, 1, 1]", rsqrt_20: "f32[1, 20, 1, 1]", arg127_1: "f32[20]", getitem_40: "f32[1, 20, 1, 1]", arg128_1: "f32[20]", arg132_1: "i64[]", getitem_43: "f32[1, 24, 1, 1]", rsqrt_21: "f32[1, 24, 1, 1]", arg133_1: "f32[24]", getitem_42: "f32[1, 24, 1, 1]", arg134_1: "f32[24]", arg138_1: "i64[]", getitem_45: "f32[1, 40, 1, 1]", rsqrt_22: "f32[1, 40, 1, 1]", arg139_1: "f32[40]", getitem_44: "f32[1, 40, 1, 1]", arg140_1: "f32[40]", arg144_1: "i64[]", getitem_47: "f32[1, 60, 1, 1]", rsqrt_23: "f32[1, 60, 1, 1]", arg145_1: "f32[60]", getitem_46: "f32[1, 60, 1, 1]", arg146_1: "f32[60]", arg150_1: "i64[]", getitem_49: "f32[1, 60, 1, 1]", arg151_1: "f32[60]", getitem_48: "f32[1, 60, 1, 1]", arg152_1: "f32[60]", arg160_1: "i64[]", getitem_51: "f32[1, 20, 1, 1]", rsqrt_25: "f32[1, 20, 1, 1]", arg161_1: "f32[20]", getitem_50: "f32[1, 20, 1, 1]", arg162_1: "f32[20]", arg166_1: "i64[]", getitem_53: "f32[1, 20, 1, 1]", rsqrt_26: "f32[1, 20, 1, 1]", arg167_1: "f32[20]", getitem_52: "f32[1, 20, 1, 1]", arg168_1: "f32[20]", arg172_1: "i64[]", getitem_55: "f32[1, 120, 1, 1]", rsqrt_27: "f32[1, 120, 1, 1]", arg173_1: "f32[120]", getitem_54: "f32[1, 120, 1, 1]", arg174_1: "f32[120]", arg178_1: "i64[]", getitem_57: "f32[1, 120, 1, 1]", arg179_1: "f32[120]", getitem_56: "f32[1, 120, 1, 1]", arg180_1: "f32[120]", arg184_1: "i64[]", getitem_59: "f32[1, 240, 1, 1]", rsqrt_29: "f32[1, 240, 1, 1]", arg185_1: "f32[240]", getitem_58: "f32[1, 240, 1, 1]", arg186_1: "f32[240]", arg190_1: "i64[]", getitem_61: "f32[1, 40, 1, 1]", rsqrt_30: "f32[1, 40, 1, 1]", arg191_1: "f32[40]", getitem_60: "f32[1, 40, 1, 1]", arg192_1: "f32[40]", arg196_1: "i64[]", getitem_63: "f32[1, 40, 1, 1]", rsqrt_31: "f32[1, 40, 1, 1]", arg197_1: "f32[40]", getitem_62: "f32[1, 40, 1, 1]", arg198_1: "f32[40]", arg202_1: "i64[]", getitem_65: "f32[1, 40, 1, 1]", rsqrt_32: "f32[1, 40, 1, 1]", arg203_1: "f32[40]", getitem_64: "f32[1, 40, 1, 1]", arg204_1: "f32[40]", arg208_1: "i64[]", getitem_67: "f32[1, 80, 1, 1]", rsqrt_33: "f32[1, 80, 1, 1]", arg209_1: "f32[80]", getitem_66: "f32[1, 80, 1, 1]", arg210_1: "f32[80]", arg214_1: "i64[]", getitem_69: "f32[1, 100, 1, 1]", rsqrt_34: "f32[1, 100, 1, 1]", arg215_1: "f32[100]", getitem_68: "f32[1, 100, 1, 1]", arg216_1: "f32[100]", arg220_1: "i64[]", getitem_71: "f32[1, 100, 1, 1]", arg221_1: "f32[100]", getitem_70: "f32[1, 100, 1, 1]", arg222_1: "f32[100]", arg226_1: "i64[]", getitem_73: "f32[1, 40, 1, 1]", rsqrt_36: "f32[1, 40, 1, 1]", arg227_1: "f32[40]", getitem_72: "f32[1, 40, 1, 1]", arg228_1: "f32[40]", arg232_1: "i64[]", getitem_75: "f32[1, 40, 1, 1]", rsqrt_37: "f32[1, 40, 1, 1]", arg233_1: "f32[40]", getitem_74: "f32[1, 40, 1, 1]", arg234_1: "f32[40]", arg238_1: "i64[]", getitem_77: "f32[1, 92, 1, 1]", rsqrt_38: "f32[1, 92, 1, 1]", arg239_1: "f32[92]", getitem_76: "f32[1, 92, 1, 1]", arg240_1: "f32[92]", arg244_1: "i64[]", getitem_79: "f32[1, 92, 1, 1]", arg245_1: "f32[92]", getitem_78: "f32[1, 92, 1, 1]", arg246_1: "f32[92]", arg250_1: "i64[]", getitem_81: "f32[1, 40, 1, 1]", rsqrt_40: "f32[1, 40, 1, 1]", arg251_1: "f32[40]", getitem_80: "f32[1, 40, 1, 1]", arg252_1: "f32[40]", arg256_1: "i64[]", getitem_83: "f32[1, 40, 1, 1]", rsqrt_41: "f32[1, 40, 1, 1]", arg257_1: "f32[40]", getitem_82: "f32[1, 40, 1, 1]", arg258_1: "f32[40]", arg262_1: "i64[]", getitem_85: "f32[1, 92, 1, 1]", rsqrt_42: "f32[1, 92, 1, 1]", arg263_1: "f32[92]", getitem_84: "f32[1, 92, 1, 1]", arg264_1: "f32[92]", arg268_1: "i64[]", getitem_87: "f32[1, 92, 1, 1]", arg269_1: "f32[92]", getitem_86: "f32[1, 92, 1, 1]", arg270_1: "f32[92]", arg274_1: "i64[]", getitem_89: "f32[1, 40, 1, 1]", rsqrt_44: "f32[1, 40, 1, 1]", arg275_1: "f32[40]", getitem_88: "f32[1, 40, 1, 1]", arg276_1: "f32[40]", arg280_1: "i64[]", getitem_91: "f32[1, 40, 1, 1]", rsqrt_45: "f32[1, 40, 1, 1]", arg281_1: "f32[40]", getitem_90: "f32[1, 40, 1, 1]", arg282_1: "f32[40]", arg286_1: "i64[]", getitem_93: "f32[1, 240, 1, 1]", rsqrt_46: "f32[1, 240, 1, 1]", arg287_1: "f32[240]", getitem_92: "f32[1, 240, 1, 1]", arg288_1: "f32[240]", arg292_1: "i64[]", getitem_95: "f32[1, 240, 1, 1]", arg293_1: "f32[240]", getitem_94: "f32[1, 240, 1, 1]", arg294_1: "f32[240]", arg302_1: "i64[]", getitem_97: "f32[1, 56, 1, 1]", rsqrt_48: "f32[1, 56, 1, 1]", arg303_1: "f32[56]", getitem_96: "f32[1, 56, 1, 1]", arg304_1: "f32[56]", arg308_1: "i64[]", getitem_99: "f32[1, 56, 1, 1]", rsqrt_49: "f32[1, 56, 1, 1]", arg309_1: "f32[56]", getitem_98: "f32[1, 56, 1, 1]", arg310_1: "f32[56]", arg314_1: "i64[]", getitem_101: "f32[1, 80, 1, 1]", rsqrt_50: "f32[1, 80, 1, 1]", arg315_1: "f32[80]", getitem_100: "f32[1, 80, 1, 1]", arg316_1: "f32[80]", arg320_1: "i64[]", getitem_103: "f32[1, 112, 1, 1]", rsqrt_51: "f32[1, 112, 1, 1]", arg321_1: "f32[112]", getitem_102: "f32[1, 112, 1, 1]", arg322_1: "f32[112]", arg326_1: "i64[]", getitem_105: "f32[1, 336, 1, 1]", rsqrt_52: "f32[1, 336, 1, 1]", arg327_1: "f32[336]", getitem_104: "f32[1, 336, 1, 1]", arg328_1: "f32[336]", arg332_1: "i64[]", getitem_107: "f32[1, 336, 1, 1]", arg333_1: "f32[336]", getitem_106: "f32[1, 336, 1, 1]", arg334_1: "f32[336]", arg342_1: "i64[]", getitem_109: "f32[1, 56, 1, 1]", rsqrt_54: "f32[1, 56, 1, 1]", arg343_1: "f32[56]", getitem_108: "f32[1, 56, 1, 1]", arg344_1: "f32[56]", arg348_1: "i64[]", getitem_111: "f32[1, 56, 1, 1]", rsqrt_55: "f32[1, 56, 1, 1]", arg349_1: "f32[56]", getitem_110: "f32[1, 56, 1, 1]", arg350_1: "f32[56]", arg354_1: "i64[]", getitem_113: "f32[1, 336, 1, 1]", rsqrt_56: "f32[1, 336, 1, 1]", arg355_1: "f32[336]", getitem_112: "f32[1, 336, 1, 1]", arg356_1: "f32[336]", arg360_1: "i64[]", getitem_115: "f32[1, 336, 1, 1]", arg361_1: "f32[336]", getitem_114: "f32[1, 336, 1, 1]", arg362_1: "f32[336]", arg366_1: "i64[]", getitem_117: "f32[1, 672, 1, 1]", arg367_1: "f32[672]", getitem_116: "f32[1, 672, 1, 1]", arg368_1: "f32[672]", arg376_1: "i64[]", getitem_119: "f32[1, 80, 1, 1]", rsqrt_59: "f32[1, 80, 1, 1]", arg377_1: "f32[80]", getitem_118: "f32[1, 80, 1, 1]", arg378_1: "f32[80]", arg382_1: "i64[]", getitem_121: "f32[1, 80, 1, 1]", rsqrt_60: "f32[1, 80, 1, 1]", arg383_1: "f32[80]", getitem_120: "f32[1, 80, 1, 1]", arg384_1: "f32[80]", arg388_1: "i64[]", getitem_123: "f32[1, 112, 1, 1]", rsqrt_61: "f32[1, 112, 1, 1]", arg389_1: "f32[112]", getitem_122: "f32[1, 112, 1, 1]", arg390_1: "f32[112]", arg394_1: "i64[]", getitem_125: "f32[1, 160, 1, 1]", rsqrt_62: "f32[1, 160, 1, 1]", arg395_1: "f32[160]", getitem_124: "f32[1, 160, 1, 1]", arg396_1: "f32[160]", arg400_1: "i64[]", getitem_127: "f32[1, 480, 1, 1]", rsqrt_63: "f32[1, 480, 1, 1]", arg401_1: "f32[480]", getitem_126: "f32[1, 480, 1, 1]", arg402_1: "f32[480]", arg406_1: "i64[]", getitem_129: "f32[1, 480, 1, 1]", arg407_1: "f32[480]", getitem_128: "f32[1, 480, 1, 1]", arg408_1: "f32[480]", arg412_1: "i64[]", getitem_131: "f32[1, 80, 1, 1]", rsqrt_65: "f32[1, 80, 1, 1]", arg413_1: "f32[80]", getitem_130: "f32[1, 80, 1, 1]", arg414_1: "f32[80]", arg418_1: "i64[]", getitem_133: "f32[1, 80, 1, 1]", rsqrt_66: "f32[1, 80, 1, 1]", arg419_1: "f32[80]", getitem_132: "f32[1, 80, 1, 1]", arg420_1: "f32[80]", arg424_1: "i64[]", getitem_135: "f32[1, 480, 1, 1]", rsqrt_67: "f32[1, 480, 1, 1]", arg425_1: "f32[480]", getitem_134: "f32[1, 480, 1, 1]", arg426_1: "f32[480]", arg430_1: "i64[]", getitem_137: "f32[1, 480, 1, 1]", arg431_1: "f32[480]", getitem_136: "f32[1, 480, 1, 1]", arg432_1: "f32[480]", arg440_1: "i64[]", getitem_139: "f32[1, 80, 1, 1]", rsqrt_69: "f32[1, 80, 1, 1]", arg441_1: "f32[80]", getitem_138: "f32[1, 80, 1, 1]", arg442_1: "f32[80]", arg446_1: "i64[]", getitem_141: "f32[1, 80, 1, 1]", rsqrt_70: "f32[1, 80, 1, 1]", arg447_1: "f32[80]", getitem_140: "f32[1, 80, 1, 1]", arg448_1: "f32[80]", arg452_1: "i64[]", getitem_143: "f32[1, 480, 1, 1]", rsqrt_71: "f32[1, 480, 1, 1]", arg453_1: "f32[480]", getitem_142: "f32[1, 480, 1, 1]", arg454_1: "f32[480]", arg458_1: "i64[]", getitem_145: "f32[1, 480, 1, 1]", arg459_1: "f32[480]", getitem_144: "f32[1, 480, 1, 1]", arg460_1: "f32[480]", arg464_1: "i64[]", getitem_147: "f32[1, 80, 1, 1]", rsqrt_73: "f32[1, 80, 1, 1]", arg465_1: "f32[80]", getitem_146: "f32[1, 80, 1, 1]", arg466_1: "f32[80]", arg470_1: "i64[]", getitem_149: "f32[1, 80, 1, 1]", rsqrt_74: "f32[1, 80, 1, 1]", arg471_1: "f32[80]", getitem_148: "f32[1, 80, 1, 1]", arg472_1: "f32[80]", arg476_1: "i64[]", getitem_151: "f32[1, 480, 1, 1]", rsqrt_75: "f32[1, 480, 1, 1]", arg477_1: "f32[480]", getitem_150: "f32[1, 480, 1, 1]", arg478_1: "f32[480]", arg482_1: "i64[]", getitem_153: "f32[1, 480, 1, 1]", arg483_1: "f32[480]", getitem_152: "f32[1, 480, 1, 1]", arg484_1: "f32[480]", arg492_1: "i64[]", getitem_155: "f32[1, 80, 1, 1]", rsqrt_77: "f32[1, 80, 1, 1]", arg493_1: "f32[80]", getitem_154: "f32[1, 80, 1, 1]", arg494_1: "f32[80]", arg498_1: "i64[]", getitem_157: "f32[1, 80, 1, 1]", rsqrt_78: "f32[1, 80, 1, 1]", arg499_1: "f32[80]", getitem_156: "f32[1, 80, 1, 1]", arg500_1: "f32[80]", arg504_1: "i64[]", getitem_159: "f32[1, 960, 1, 1]", arg505_1: "f32[960]", getitem_158: "f32[1, 960, 1, 1]", arg506_1: "f32[960]", convolution_94: "f16[8, 1280, 1, 1]", arg512_1: "f32[1000]", arg511_1: "f32[1000, 1280]", _shape_param_0):
        # No stacktrace found for following nodes
        add_tensor: "i64[]" = torch.ops.aten.add.Tensor(arg2_1, 1)
        squeeze_dims: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        squeeze_dims_1: "f32[16]" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
        mul_tensor: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_dims, 0.1)
        mul_tensor_1: "f32[16]" = torch.ops.aten.mul.Tensor(arg3_1, 0.9)
        add_tensor_1: "f32[16]" = torch.ops.aten.add.Tensor(mul_tensor, mul_tensor_1);  mul_tensor = mul_tensor_1 = None
        squeeze_dims_2: "f32[16]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_tensor_2: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_dims_2, 1.00000996502277);  squeeze_dims_2 = None
        mul_tensor_3: "f32[16]" = torch.ops.aten.mul.Tensor(mul_tensor_2, 0.1);  mul_tensor_2 = None
        mul_tensor_4: "f32[16]" = torch.ops.aten.mul.Tensor(arg4_1, 0.9)
        add_tensor_2: "f32[16]" = torch.ops.aten.add.Tensor(mul_tensor_3, mul_tensor_4);  mul_tensor_3 = mul_tensor_4 = None
        add_tensor_3: "i64[]" = torch.ops.aten.add.Tensor(arg8_1, 1)
        squeeze_dims_3: "f32[8]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
        squeeze_dims_4: "f32[8]" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
        mul_tensor_5: "f32[8]" = torch.ops.aten.mul.Tensor(squeeze_dims_3, 0.1)
        mul_tensor_6: "f32[8]" = torch.ops.aten.mul.Tensor(arg9_1, 0.9)
        add_tensor_4: "f32[8]" = torch.ops.aten.add.Tensor(mul_tensor_5, mul_tensor_6);  mul_tensor_5 = mul_tensor_6 = None
        squeeze_dims_5: "f32[8]" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
        mul_tensor_7: "f32[8]" = torch.ops.aten.mul.Tensor(squeeze_dims_5, 1.00000996502277);  squeeze_dims_5 = None
        mul_tensor_8: "f32[8]" = torch.ops.aten.mul.Tensor(mul_tensor_7, 0.1);  mul_tensor_7 = None
        mul_tensor_9: "f32[8]" = torch.ops.aten.mul.Tensor(arg10_1, 0.9)
        add_tensor_5: "f32[8]" = torch.ops.aten.add.Tensor(mul_tensor_8, mul_tensor_9);  mul_tensor_8 = mul_tensor_9 = None
        add_tensor_6: "i64[]" = torch.ops.aten.add.Tensor(arg14_1, 1)
        squeeze_dims_6: "f32[8]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
        mul_tensor_10: "f32[8]" = torch.ops.aten.mul.Tensor(squeeze_dims_6, 0.1);  squeeze_dims_6 = None
        mul_tensor_11: "f32[8]" = torch.ops.aten.mul.Tensor(arg15_1, 0.9)
        add_tensor_7: "f32[8]" = torch.ops.aten.add.Tensor(mul_tensor_10, mul_tensor_11);  mul_tensor_10 = mul_tensor_11 = None
        squeeze_dims_7: "f32[8]" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
        mul_tensor_12: "f32[8]" = torch.ops.aten.mul.Tensor(squeeze_dims_7, 1.00000996502277);  squeeze_dims_7 = None
        mul_tensor_13: "f32[8]" = torch.ops.aten.mul.Tensor(mul_tensor_12, 0.1);  mul_tensor_12 = None
        mul_tensor_14: "f32[8]" = torch.ops.aten.mul.Tensor(arg16_1, 0.9)
        add_tensor_8: "f32[8]" = torch.ops.aten.add.Tensor(mul_tensor_13, mul_tensor_14);  mul_tensor_13 = mul_tensor_14 = None
        add_tensor_9: "i64[]" = torch.ops.aten.add.Tensor(arg20_1, 1)
        squeeze_dims_8: "f32[8]" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2, 3]);  getitem_7 = None
        squeeze_dims_9: "f32[8]" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
        mul_tensor_15: "f32[8]" = torch.ops.aten.mul.Tensor(squeeze_dims_8, 0.1)
        mul_tensor_16: "f32[8]" = torch.ops.aten.mul.Tensor(arg21_1, 0.9)
        add_tensor_10: "f32[8]" = torch.ops.aten.add.Tensor(mul_tensor_15, mul_tensor_16);  mul_tensor_15 = mul_tensor_16 = None
        squeeze_dims_10: "f32[8]" = torch.ops.aten.squeeze.dims(getitem_6, [0, 2, 3]);  getitem_6 = None
        mul_tensor_17: "f32[8]" = torch.ops.aten.mul.Tensor(squeeze_dims_10, 1.00000996502277);  squeeze_dims_10 = None
        mul_tensor_18: "f32[8]" = torch.ops.aten.mul.Tensor(mul_tensor_17, 0.1);  mul_tensor_17 = None
        mul_tensor_19: "f32[8]" = torch.ops.aten.mul.Tensor(arg22_1, 0.9)
        add_tensor_11: "f32[8]" = torch.ops.aten.add.Tensor(mul_tensor_18, mul_tensor_19);  mul_tensor_18 = mul_tensor_19 = None
        add_tensor_12: "i64[]" = torch.ops.aten.add.Tensor(arg26_1, 1)
        squeeze_dims_11: "f32[8]" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3]);  getitem_9 = None
        squeeze_dims_12: "f32[8]" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
        mul_tensor_20: "f32[8]" = torch.ops.aten.mul.Tensor(squeeze_dims_11, 0.1)
        mul_tensor_21: "f32[8]" = torch.ops.aten.mul.Tensor(arg27_1, 0.9)
        add_tensor_13: "f32[8]" = torch.ops.aten.add.Tensor(mul_tensor_20, mul_tensor_21);  mul_tensor_20 = mul_tensor_21 = None
        squeeze_dims_13: "f32[8]" = torch.ops.aten.squeeze.dims(getitem_8, [0, 2, 3]);  getitem_8 = None
        mul_tensor_22: "f32[8]" = torch.ops.aten.mul.Tensor(squeeze_dims_13, 1.00000996502277);  squeeze_dims_13 = None
        mul_tensor_23: "f32[8]" = torch.ops.aten.mul.Tensor(mul_tensor_22, 0.1);  mul_tensor_22 = None
        mul_tensor_24: "f32[8]" = torch.ops.aten.mul.Tensor(arg28_1, 0.9)
        add_tensor_14: "f32[8]" = torch.ops.aten.add.Tensor(mul_tensor_23, mul_tensor_24);  mul_tensor_23 = mul_tensor_24 = None
        add_tensor_15: "i64[]" = torch.ops.aten.add.Tensor(arg32_1, 1)
        squeeze_dims_14: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3]);  getitem_11 = None
        squeeze_dims_15: "f32[24]" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
        mul_tensor_25: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_dims_14, 0.1)
        mul_tensor_26: "f32[24]" = torch.ops.aten.mul.Tensor(arg33_1, 0.9)
        add_tensor_16: "f32[24]" = torch.ops.aten.add.Tensor(mul_tensor_25, mul_tensor_26);  mul_tensor_25 = mul_tensor_26 = None
        squeeze_dims_16: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_10, [0, 2, 3]);  getitem_10 = None
        mul_tensor_27: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_dims_16, 1.00000996502277);  squeeze_dims_16 = None
        mul_tensor_28: "f32[24]" = torch.ops.aten.mul.Tensor(mul_tensor_27, 0.1);  mul_tensor_27 = None
        mul_tensor_29: "f32[24]" = torch.ops.aten.mul.Tensor(arg34_1, 0.9)
        add_tensor_17: "f32[24]" = torch.ops.aten.add.Tensor(mul_tensor_28, mul_tensor_29);  mul_tensor_28 = mul_tensor_29 = None
        add_tensor_18: "i64[]" = torch.ops.aten.add.Tensor(arg38_1, 1)
        squeeze_dims_17: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
        mul_tensor_30: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_dims_17, 0.1);  squeeze_dims_17 = None
        mul_tensor_31: "f32[24]" = torch.ops.aten.mul.Tensor(arg39_1, 0.9)
        add_tensor_19: "f32[24]" = torch.ops.aten.add.Tensor(mul_tensor_30, mul_tensor_31);  mul_tensor_30 = mul_tensor_31 = None
        squeeze_dims_18: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_12, [0, 2, 3]);  getitem_12 = None
        mul_tensor_32: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_dims_18, 1.00000996502277);  squeeze_dims_18 = None
        mul_tensor_33: "f32[24]" = torch.ops.aten.mul.Tensor(mul_tensor_32, 0.1);  mul_tensor_32 = None
        mul_tensor_34: "f32[24]" = torch.ops.aten.mul.Tensor(arg40_1, 0.9)
        add_tensor_20: "f32[24]" = torch.ops.aten.add.Tensor(mul_tensor_33, mul_tensor_34);  mul_tensor_33 = mul_tensor_34 = None
        add_tensor_21: "i64[]" = torch.ops.aten.add.Tensor(arg44_1, 1)
        squeeze_dims_19: "f32[48]" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3]);  getitem_15 = None
        squeeze_dims_20: "f32[48]" = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2, 3]);  rsqrt_7 = None
        mul_tensor_35: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_dims_19, 0.1)
        mul_tensor_36: "f32[48]" = torch.ops.aten.mul.Tensor(arg45_1, 0.9)
        add_tensor_22: "f32[48]" = torch.ops.aten.add.Tensor(mul_tensor_35, mul_tensor_36);  mul_tensor_35 = mul_tensor_36 = None
        squeeze_dims_21: "f32[48]" = torch.ops.aten.squeeze.dims(getitem_14, [0, 2, 3]);  getitem_14 = None
        mul_tensor_37: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_dims_21, 1.0000398612827361);  squeeze_dims_21 = None
        mul_tensor_38: "f32[48]" = torch.ops.aten.mul.Tensor(mul_tensor_37, 0.1);  mul_tensor_37 = None
        mul_tensor_39: "f32[48]" = torch.ops.aten.mul.Tensor(arg46_1, 0.9)
        add_tensor_23: "f32[48]" = torch.ops.aten.add.Tensor(mul_tensor_38, mul_tensor_39);  mul_tensor_38 = mul_tensor_39 = None
        add_tensor_24: "i64[]" = torch.ops.aten.add.Tensor(arg50_1, 1)
        squeeze_dims_22: "f32[12]" = torch.ops.aten.squeeze.dims(getitem_17, [0, 2, 3]);  getitem_17 = None
        squeeze_dims_23: "f32[12]" = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2, 3]);  rsqrt_8 = None
        mul_tensor_40: "f32[12]" = torch.ops.aten.mul.Tensor(squeeze_dims_22, 0.1)
        mul_tensor_41: "f32[12]" = torch.ops.aten.mul.Tensor(arg51_1, 0.9)
        add_tensor_25: "f32[12]" = torch.ops.aten.add.Tensor(mul_tensor_40, mul_tensor_41);  mul_tensor_40 = mul_tensor_41 = None
        squeeze_dims_24: "f32[12]" = torch.ops.aten.squeeze.dims(getitem_16, [0, 2, 3]);  getitem_16 = None
        mul_tensor_42: "f32[12]" = torch.ops.aten.mul.Tensor(squeeze_dims_24, 1.0000398612827361);  squeeze_dims_24 = None
        mul_tensor_43: "f32[12]" = torch.ops.aten.mul.Tensor(mul_tensor_42, 0.1);  mul_tensor_42 = None
        mul_tensor_44: "f32[12]" = torch.ops.aten.mul.Tensor(arg52_1, 0.9)
        add_tensor_26: "f32[12]" = torch.ops.aten.add.Tensor(mul_tensor_43, mul_tensor_44);  mul_tensor_43 = mul_tensor_44 = None
        add_tensor_27: "i64[]" = torch.ops.aten.add.Tensor(arg56_1, 1)
        squeeze_dims_25: "f32[12]" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3]);  getitem_19 = None
        squeeze_dims_26: "f32[12]" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
        mul_tensor_45: "f32[12]" = torch.ops.aten.mul.Tensor(squeeze_dims_25, 0.1)
        mul_tensor_46: "f32[12]" = torch.ops.aten.mul.Tensor(arg57_1, 0.9)
        add_tensor_28: "f32[12]" = torch.ops.aten.add.Tensor(mul_tensor_45, mul_tensor_46);  mul_tensor_45 = mul_tensor_46 = None
        squeeze_dims_27: "f32[12]" = torch.ops.aten.squeeze.dims(getitem_18, [0, 2, 3]);  getitem_18 = None
        mul_tensor_47: "f32[12]" = torch.ops.aten.mul.Tensor(squeeze_dims_27, 1.0000398612827361);  squeeze_dims_27 = None
        mul_tensor_48: "f32[12]" = torch.ops.aten.mul.Tensor(mul_tensor_47, 0.1);  mul_tensor_47 = None
        mul_tensor_49: "f32[12]" = torch.ops.aten.mul.Tensor(arg58_1, 0.9)
        add_tensor_29: "f32[12]" = torch.ops.aten.add.Tensor(mul_tensor_48, mul_tensor_49);  mul_tensor_48 = mul_tensor_49 = None
        add_tensor_30: "i64[]" = torch.ops.aten.add.Tensor(arg62_1, 1)
        squeeze_dims_28: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
        squeeze_dims_29: "f32[16]" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
        mul_tensor_50: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_dims_28, 0.1)
        mul_tensor_51: "f32[16]" = torch.ops.aten.mul.Tensor(arg63_1, 0.9)
        add_tensor_31: "f32[16]" = torch.ops.aten.add.Tensor(mul_tensor_50, mul_tensor_51);  mul_tensor_50 = mul_tensor_51 = None
        squeeze_dims_30: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_20, [0, 2, 3]);  getitem_20 = None
        mul_tensor_52: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_dims_30, 1.0000398612827361);  squeeze_dims_30 = None
        mul_tensor_53: "f32[16]" = torch.ops.aten.mul.Tensor(mul_tensor_52, 0.1);  mul_tensor_52 = None
        mul_tensor_54: "f32[16]" = torch.ops.aten.mul.Tensor(arg64_1, 0.9)
        add_tensor_32: "f32[16]" = torch.ops.aten.add.Tensor(mul_tensor_53, mul_tensor_54);  mul_tensor_53 = mul_tensor_54 = None
        add_tensor_33: "i64[]" = torch.ops.aten.add.Tensor(arg68_1, 1)
        squeeze_dims_31: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_23, [0, 2, 3]);  getitem_23 = None
        squeeze_dims_32: "f32[24]" = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2, 3]);  rsqrt_11 = None
        mul_tensor_55: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_dims_31, 0.1)
        mul_tensor_56: "f32[24]" = torch.ops.aten.mul.Tensor(arg69_1, 0.9)
        add_tensor_34: "f32[24]" = torch.ops.aten.add.Tensor(mul_tensor_55, mul_tensor_56);  mul_tensor_55 = mul_tensor_56 = None
        squeeze_dims_33: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_22, [0, 2, 3]);  getitem_22 = None
        mul_tensor_57: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_dims_33, 1.0000398612827361);  squeeze_dims_33 = None
        mul_tensor_58: "f32[24]" = torch.ops.aten.mul.Tensor(mul_tensor_57, 0.1);  mul_tensor_57 = None
        mul_tensor_59: "f32[24]" = torch.ops.aten.mul.Tensor(arg70_1, 0.9)
        add_tensor_35: "f32[24]" = torch.ops.aten.add.Tensor(mul_tensor_58, mul_tensor_59);  mul_tensor_58 = mul_tensor_59 = None
        add_tensor_36: "i64[]" = torch.ops.aten.add.Tensor(arg74_1, 1)
        squeeze_dims_34: "f32[36]" = torch.ops.aten.squeeze.dims(getitem_25, [0, 2, 3]);  getitem_25 = None
        squeeze_dims_35: "f32[36]" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
        mul_tensor_60: "f32[36]" = torch.ops.aten.mul.Tensor(squeeze_dims_34, 0.1)
        mul_tensor_61: "f32[36]" = torch.ops.aten.mul.Tensor(arg75_1, 0.9)
        add_tensor_37: "f32[36]" = torch.ops.aten.add.Tensor(mul_tensor_60, mul_tensor_61);  mul_tensor_60 = mul_tensor_61 = None
        squeeze_dims_36: "f32[36]" = torch.ops.aten.squeeze.dims(getitem_24, [0, 2, 3]);  getitem_24 = None
        mul_tensor_62: "f32[36]" = torch.ops.aten.mul.Tensor(squeeze_dims_36, 1.0000398612827361);  squeeze_dims_36 = None
        mul_tensor_63: "f32[36]" = torch.ops.aten.mul.Tensor(mul_tensor_62, 0.1);  mul_tensor_62 = None
        mul_tensor_64: "f32[36]" = torch.ops.aten.mul.Tensor(arg76_1, 0.9)
        add_tensor_38: "f32[36]" = torch.ops.aten.add.Tensor(mul_tensor_63, mul_tensor_64);  mul_tensor_63 = mul_tensor_64 = None
        add_tensor_39: "i64[]" = torch.ops.aten.add.Tensor(arg80_1, 1)
        squeeze_dims_37: "f32[36]" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3]);  getitem_27 = None
        mul_tensor_65: "f32[36]" = torch.ops.aten.mul.Tensor(squeeze_dims_37, 0.1);  squeeze_dims_37 = None
        mul_tensor_66: "f32[36]" = torch.ops.aten.mul.Tensor(arg81_1, 0.9)
        add_tensor_40: "f32[36]" = torch.ops.aten.add.Tensor(mul_tensor_65, mul_tensor_66);  mul_tensor_65 = mul_tensor_66 = None
        squeeze_dims_38: "f32[36]" = torch.ops.aten.squeeze.dims(getitem_26, [0, 2, 3]);  getitem_26 = None
        mul_tensor_67: "f32[36]" = torch.ops.aten.mul.Tensor(squeeze_dims_38, 1.0000398612827361);  squeeze_dims_38 = None
        mul_tensor_68: "f32[36]" = torch.ops.aten.mul.Tensor(mul_tensor_67, 0.1);  mul_tensor_67 = None
        mul_tensor_69: "f32[36]" = torch.ops.aten.mul.Tensor(arg82_1, 0.9)
        add_tensor_41: "f32[36]" = torch.ops.aten.add.Tensor(mul_tensor_68, mul_tensor_69);  mul_tensor_68 = mul_tensor_69 = None
        add_tensor_42: "i64[]" = torch.ops.aten.add.Tensor(arg86_1, 1)
        squeeze_dims_39: "f32[12]" = torch.ops.aten.squeeze.dims(getitem_29, [0, 2, 3]);  getitem_29 = None
        squeeze_dims_40: "f32[12]" = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2, 3]);  rsqrt_14 = None
        mul_tensor_70: "f32[12]" = torch.ops.aten.mul.Tensor(squeeze_dims_39, 0.1)
        mul_tensor_71: "f32[12]" = torch.ops.aten.mul.Tensor(arg87_1, 0.9)
        add_tensor_43: "f32[12]" = torch.ops.aten.add.Tensor(mul_tensor_70, mul_tensor_71);  mul_tensor_70 = mul_tensor_71 = None
        squeeze_dims_41: "f32[12]" = torch.ops.aten.squeeze.dims(getitem_28, [0, 2, 3]);  getitem_28 = None
        mul_tensor_72: "f32[12]" = torch.ops.aten.mul.Tensor(squeeze_dims_41, 1.0000398612827361);  squeeze_dims_41 = None
        mul_tensor_73: "f32[12]" = torch.ops.aten.mul.Tensor(mul_tensor_72, 0.1);  mul_tensor_72 = None
        mul_tensor_74: "f32[12]" = torch.ops.aten.mul.Tensor(arg88_1, 0.9)
        add_tensor_44: "f32[12]" = torch.ops.aten.add.Tensor(mul_tensor_73, mul_tensor_74);  mul_tensor_73 = mul_tensor_74 = None
        add_tensor_45: "i64[]" = torch.ops.aten.add.Tensor(arg92_1, 1)
        squeeze_dims_42: "f32[12]" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2, 3]);  getitem_31 = None
        squeeze_dims_43: "f32[12]" = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2, 3]);  rsqrt_15 = None
        mul_tensor_75: "f32[12]" = torch.ops.aten.mul.Tensor(squeeze_dims_42, 0.1)
        mul_tensor_76: "f32[12]" = torch.ops.aten.mul.Tensor(arg93_1, 0.9)
        add_tensor_46: "f32[12]" = torch.ops.aten.add.Tensor(mul_tensor_75, mul_tensor_76);  mul_tensor_75 = mul_tensor_76 = None
        squeeze_dims_44: "f32[12]" = torch.ops.aten.squeeze.dims(getitem_30, [0, 2, 3]);  getitem_30 = None
        mul_tensor_77: "f32[12]" = torch.ops.aten.mul.Tensor(squeeze_dims_44, 1.0000398612827361);  squeeze_dims_44 = None
        mul_tensor_78: "f32[12]" = torch.ops.aten.mul.Tensor(mul_tensor_77, 0.1);  mul_tensor_77 = None
        mul_tensor_79: "f32[12]" = torch.ops.aten.mul.Tensor(arg94_1, 0.9)
        add_tensor_47: "f32[12]" = torch.ops.aten.add.Tensor(mul_tensor_78, mul_tensor_79);  mul_tensor_78 = mul_tensor_79 = None
        add_tensor_48: "i64[]" = torch.ops.aten.add.Tensor(arg98_1, 1)
        squeeze_dims_45: "f32[36]" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3]);  getitem_33 = None
        squeeze_dims_46: "f32[36]" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2, 3]);  rsqrt_16 = None
        mul_tensor_80: "f32[36]" = torch.ops.aten.mul.Tensor(squeeze_dims_45, 0.1)
        mul_tensor_81: "f32[36]" = torch.ops.aten.mul.Tensor(arg99_1, 0.9)
        add_tensor_49: "f32[36]" = torch.ops.aten.add.Tensor(mul_tensor_80, mul_tensor_81);  mul_tensor_80 = mul_tensor_81 = None
        squeeze_dims_47: "f32[36]" = torch.ops.aten.squeeze.dims(getitem_32, [0, 2, 3]);  getitem_32 = None
        mul_tensor_82: "f32[36]" = torch.ops.aten.mul.Tensor(squeeze_dims_47, 1.0000398612827361);  squeeze_dims_47 = None
        mul_tensor_83: "f32[36]" = torch.ops.aten.mul.Tensor(mul_tensor_82, 0.1);  mul_tensor_82 = None
        mul_tensor_84: "f32[36]" = torch.ops.aten.mul.Tensor(arg100_1, 0.9)
        add_tensor_50: "f32[36]" = torch.ops.aten.add.Tensor(mul_tensor_83, mul_tensor_84);  mul_tensor_83 = mul_tensor_84 = None
        add_tensor_51: "i64[]" = torch.ops.aten.add.Tensor(arg104_1, 1)
        squeeze_dims_48: "f32[36]" = torch.ops.aten.squeeze.dims(getitem_35, [0, 2, 3]);  getitem_35 = None
        mul_tensor_85: "f32[36]" = torch.ops.aten.mul.Tensor(squeeze_dims_48, 0.1);  squeeze_dims_48 = None
        mul_tensor_86: "f32[36]" = torch.ops.aten.mul.Tensor(arg105_1, 0.9)
        add_tensor_52: "f32[36]" = torch.ops.aten.add.Tensor(mul_tensor_85, mul_tensor_86);  mul_tensor_85 = mul_tensor_86 = None
        squeeze_dims_49: "f32[36]" = torch.ops.aten.squeeze.dims(getitem_34, [0, 2, 3]);  getitem_34 = None
        mul_tensor_87: "f32[36]" = torch.ops.aten.mul.Tensor(squeeze_dims_49, 1.0000398612827361);  squeeze_dims_49 = None
        mul_tensor_88: "f32[36]" = torch.ops.aten.mul.Tensor(mul_tensor_87, 0.1);  mul_tensor_87 = None
        mul_tensor_89: "f32[36]" = torch.ops.aten.mul.Tensor(arg106_1, 0.9)
        add_tensor_53: "f32[36]" = torch.ops.aten.add.Tensor(mul_tensor_88, mul_tensor_89);  mul_tensor_88 = mul_tensor_89 = None
        add_tensor_54: "i64[]" = torch.ops.aten.add.Tensor(arg110_1, 1)
        squeeze_dims_50: "f32[72]" = torch.ops.aten.squeeze.dims(getitem_37, [0, 2, 3]);  getitem_37 = None
        mul_tensor_90: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_dims_50, 0.1);  squeeze_dims_50 = None
        mul_tensor_91: "f32[72]" = torch.ops.aten.mul.Tensor(arg111_1, 0.9)
        add_tensor_55: "f32[72]" = torch.ops.aten.add.Tensor(mul_tensor_90, mul_tensor_91);  mul_tensor_90 = mul_tensor_91 = None
        squeeze_dims_51: "f32[72]" = torch.ops.aten.squeeze.dims(getitem_36, [0, 2, 3]);  getitem_36 = None
        mul_tensor_92: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_dims_51, 1.0001594642002871);  squeeze_dims_51 = None
        mul_tensor_93: "f32[72]" = torch.ops.aten.mul.Tensor(mul_tensor_92, 0.1);  mul_tensor_92 = None
        mul_tensor_94: "f32[72]" = torch.ops.aten.mul.Tensor(arg112_1, 0.9)
        add_tensor_56: "f32[72]" = torch.ops.aten.add.Tensor(mul_tensor_93, mul_tensor_94);  mul_tensor_93 = mul_tensor_94 = None
        add_tensor_57: "i64[]" = torch.ops.aten.add.Tensor(arg120_1, 1)
        squeeze_dims_52: "f32[20]" = torch.ops.aten.squeeze.dims(getitem_39, [0, 2, 3]);  getitem_39 = None
        squeeze_dims_53: "f32[20]" = torch.ops.aten.squeeze.dims(rsqrt_19, [0, 2, 3]);  rsqrt_19 = None
        mul_tensor_95: "f32[20]" = torch.ops.aten.mul.Tensor(squeeze_dims_52, 0.1)
        mul_tensor_96: "f32[20]" = torch.ops.aten.mul.Tensor(arg121_1, 0.9)
        add_tensor_58: "f32[20]" = torch.ops.aten.add.Tensor(mul_tensor_95, mul_tensor_96);  mul_tensor_95 = mul_tensor_96 = None
        squeeze_dims_54: "f32[20]" = torch.ops.aten.squeeze.dims(getitem_38, [0, 2, 3]);  getitem_38 = None
        mul_tensor_97: "f32[20]" = torch.ops.aten.mul.Tensor(squeeze_dims_54, 1.0001594642002871);  squeeze_dims_54 = None
        mul_tensor_98: "f32[20]" = torch.ops.aten.mul.Tensor(mul_tensor_97, 0.1);  mul_tensor_97 = None
        mul_tensor_99: "f32[20]" = torch.ops.aten.mul.Tensor(arg122_1, 0.9)
        add_tensor_59: "f32[20]" = torch.ops.aten.add.Tensor(mul_tensor_98, mul_tensor_99);  mul_tensor_98 = mul_tensor_99 = None
        add_tensor_60: "i64[]" = torch.ops.aten.add.Tensor(arg126_1, 1)
        squeeze_dims_55: "f32[20]" = torch.ops.aten.squeeze.dims(getitem_41, [0, 2, 3]);  getitem_41 = None
        squeeze_dims_56: "f32[20]" = torch.ops.aten.squeeze.dims(rsqrt_20, [0, 2, 3]);  rsqrt_20 = None
        mul_tensor_100: "f32[20]" = torch.ops.aten.mul.Tensor(squeeze_dims_55, 0.1)
        mul_tensor_101: "f32[20]" = torch.ops.aten.mul.Tensor(arg127_1, 0.9)
        add_tensor_61: "f32[20]" = torch.ops.aten.add.Tensor(mul_tensor_100, mul_tensor_101);  mul_tensor_100 = mul_tensor_101 = None
        squeeze_dims_57: "f32[20]" = torch.ops.aten.squeeze.dims(getitem_40, [0, 2, 3]);  getitem_40 = None
        mul_tensor_102: "f32[20]" = torch.ops.aten.mul.Tensor(squeeze_dims_57, 1.0001594642002871);  squeeze_dims_57 = None
        mul_tensor_103: "f32[20]" = torch.ops.aten.mul.Tensor(mul_tensor_102, 0.1);  mul_tensor_102 = None
        mul_tensor_104: "f32[20]" = torch.ops.aten.mul.Tensor(arg128_1, 0.9)
        add_tensor_62: "f32[20]" = torch.ops.aten.add.Tensor(mul_tensor_103, mul_tensor_104);  mul_tensor_103 = mul_tensor_104 = None
        add_tensor_63: "i64[]" = torch.ops.aten.add.Tensor(arg132_1, 1)
        squeeze_dims_58: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_43, [0, 2, 3]);  getitem_43 = None
        squeeze_dims_59: "f32[24]" = torch.ops.aten.squeeze.dims(rsqrt_21, [0, 2, 3]);  rsqrt_21 = None
        mul_tensor_105: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_dims_58, 0.1)
        mul_tensor_106: "f32[24]" = torch.ops.aten.mul.Tensor(arg133_1, 0.9)
        add_tensor_64: "f32[24]" = torch.ops.aten.add.Tensor(mul_tensor_105, mul_tensor_106);  mul_tensor_105 = mul_tensor_106 = None
        squeeze_dims_60: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_42, [0, 2, 3]);  getitem_42 = None
        mul_tensor_107: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_dims_60, 1.0001594642002871);  squeeze_dims_60 = None
        mul_tensor_108: "f32[24]" = torch.ops.aten.mul.Tensor(mul_tensor_107, 0.1);  mul_tensor_107 = None
        mul_tensor_109: "f32[24]" = torch.ops.aten.mul.Tensor(arg134_1, 0.9)
        add_tensor_65: "f32[24]" = torch.ops.aten.add.Tensor(mul_tensor_108, mul_tensor_109);  mul_tensor_108 = mul_tensor_109 = None
        add_tensor_66: "i64[]" = torch.ops.aten.add.Tensor(arg138_1, 1)
        squeeze_dims_61: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_45, [0, 2, 3]);  getitem_45 = None
        squeeze_dims_62: "f32[40]" = torch.ops.aten.squeeze.dims(rsqrt_22, [0, 2, 3]);  rsqrt_22 = None
        mul_tensor_110: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_dims_61, 0.1)
        mul_tensor_111: "f32[40]" = torch.ops.aten.mul.Tensor(arg139_1, 0.9)
        add_tensor_67: "f32[40]" = torch.ops.aten.add.Tensor(mul_tensor_110, mul_tensor_111);  mul_tensor_110 = mul_tensor_111 = None
        squeeze_dims_63: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_44, [0, 2, 3]);  getitem_44 = None
        mul_tensor_112: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_dims_63, 1.0001594642002871);  squeeze_dims_63 = None
        mul_tensor_113: "f32[40]" = torch.ops.aten.mul.Tensor(mul_tensor_112, 0.1);  mul_tensor_112 = None
        mul_tensor_114: "f32[40]" = torch.ops.aten.mul.Tensor(arg140_1, 0.9)
        add_tensor_68: "f32[40]" = torch.ops.aten.add.Tensor(mul_tensor_113, mul_tensor_114);  mul_tensor_113 = mul_tensor_114 = None
        add_tensor_69: "i64[]" = torch.ops.aten.add.Tensor(arg144_1, 1)
        squeeze_dims_64: "f32[60]" = torch.ops.aten.squeeze.dims(getitem_47, [0, 2, 3]);  getitem_47 = None
        squeeze_dims_65: "f32[60]" = torch.ops.aten.squeeze.dims(rsqrt_23, [0, 2, 3]);  rsqrt_23 = None
        mul_tensor_115: "f32[60]" = torch.ops.aten.mul.Tensor(squeeze_dims_64, 0.1)
        mul_tensor_116: "f32[60]" = torch.ops.aten.mul.Tensor(arg145_1, 0.9)
        add_tensor_70: "f32[60]" = torch.ops.aten.add.Tensor(mul_tensor_115, mul_tensor_116);  mul_tensor_115 = mul_tensor_116 = None
        squeeze_dims_66: "f32[60]" = torch.ops.aten.squeeze.dims(getitem_46, [0, 2, 3]);  getitem_46 = None
        mul_tensor_117: "f32[60]" = torch.ops.aten.mul.Tensor(squeeze_dims_66, 1.0001594642002871);  squeeze_dims_66 = None
        mul_tensor_118: "f32[60]" = torch.ops.aten.mul.Tensor(mul_tensor_117, 0.1);  mul_tensor_117 = None
        mul_tensor_119: "f32[60]" = torch.ops.aten.mul.Tensor(arg146_1, 0.9)
        add_tensor_71: "f32[60]" = torch.ops.aten.add.Tensor(mul_tensor_118, mul_tensor_119);  mul_tensor_118 = mul_tensor_119 = None
        add_tensor_72: "i64[]" = torch.ops.aten.add.Tensor(arg150_1, 1)
        squeeze_dims_67: "f32[60]" = torch.ops.aten.squeeze.dims(getitem_49, [0, 2, 3]);  getitem_49 = None
        mul_tensor_120: "f32[60]" = torch.ops.aten.mul.Tensor(squeeze_dims_67, 0.1);  squeeze_dims_67 = None
        mul_tensor_121: "f32[60]" = torch.ops.aten.mul.Tensor(arg151_1, 0.9)
        add_tensor_73: "f32[60]" = torch.ops.aten.add.Tensor(mul_tensor_120, mul_tensor_121);  mul_tensor_120 = mul_tensor_121 = None
        squeeze_dims_68: "f32[60]" = torch.ops.aten.squeeze.dims(getitem_48, [0, 2, 3]);  getitem_48 = None
        mul_tensor_122: "f32[60]" = torch.ops.aten.mul.Tensor(squeeze_dims_68, 1.0001594642002871);  squeeze_dims_68 = None
        mul_tensor_123: "f32[60]" = torch.ops.aten.mul.Tensor(mul_tensor_122, 0.1);  mul_tensor_122 = None
        mul_tensor_124: "f32[60]" = torch.ops.aten.mul.Tensor(arg152_1, 0.9)
        add_tensor_74: "f32[60]" = torch.ops.aten.add.Tensor(mul_tensor_123, mul_tensor_124);  mul_tensor_123 = mul_tensor_124 = None
        add_tensor_75: "i64[]" = torch.ops.aten.add.Tensor(arg160_1, 1)
        squeeze_dims_69: "f32[20]" = torch.ops.aten.squeeze.dims(getitem_51, [0, 2, 3]);  getitem_51 = None
        squeeze_dims_70: "f32[20]" = torch.ops.aten.squeeze.dims(rsqrt_25, [0, 2, 3]);  rsqrt_25 = None
        mul_tensor_125: "f32[20]" = torch.ops.aten.mul.Tensor(squeeze_dims_69, 0.1)
        mul_tensor_126: "f32[20]" = torch.ops.aten.mul.Tensor(arg161_1, 0.9)
        add_tensor_76: "f32[20]" = torch.ops.aten.add.Tensor(mul_tensor_125, mul_tensor_126);  mul_tensor_125 = mul_tensor_126 = None
        squeeze_dims_71: "f32[20]" = torch.ops.aten.squeeze.dims(getitem_50, [0, 2, 3]);  getitem_50 = None
        mul_tensor_127: "f32[20]" = torch.ops.aten.mul.Tensor(squeeze_dims_71, 1.0001594642002871);  squeeze_dims_71 = None
        mul_tensor_128: "f32[20]" = torch.ops.aten.mul.Tensor(mul_tensor_127, 0.1);  mul_tensor_127 = None
        mul_tensor_129: "f32[20]" = torch.ops.aten.mul.Tensor(arg162_1, 0.9)
        add_tensor_77: "f32[20]" = torch.ops.aten.add.Tensor(mul_tensor_128, mul_tensor_129);  mul_tensor_128 = mul_tensor_129 = None
        add_tensor_78: "i64[]" = torch.ops.aten.add.Tensor(arg166_1, 1)
        squeeze_dims_72: "f32[20]" = torch.ops.aten.squeeze.dims(getitem_53, [0, 2, 3]);  getitem_53 = None
        squeeze_dims_73: "f32[20]" = torch.ops.aten.squeeze.dims(rsqrt_26, [0, 2, 3]);  rsqrt_26 = None
        mul_tensor_130: "f32[20]" = torch.ops.aten.mul.Tensor(squeeze_dims_72, 0.1)
        mul_tensor_131: "f32[20]" = torch.ops.aten.mul.Tensor(arg167_1, 0.9)
        add_tensor_79: "f32[20]" = torch.ops.aten.add.Tensor(mul_tensor_130, mul_tensor_131);  mul_tensor_130 = mul_tensor_131 = None
        squeeze_dims_74: "f32[20]" = torch.ops.aten.squeeze.dims(getitem_52, [0, 2, 3]);  getitem_52 = None
        mul_tensor_132: "f32[20]" = torch.ops.aten.mul.Tensor(squeeze_dims_74, 1.0001594642002871);  squeeze_dims_74 = None
        mul_tensor_133: "f32[20]" = torch.ops.aten.mul.Tensor(mul_tensor_132, 0.1);  mul_tensor_132 = None
        mul_tensor_134: "f32[20]" = torch.ops.aten.mul.Tensor(arg168_1, 0.9)
        add_tensor_80: "f32[20]" = torch.ops.aten.add.Tensor(mul_tensor_133, mul_tensor_134);  mul_tensor_133 = mul_tensor_134 = None
        add_tensor_81: "i64[]" = torch.ops.aten.add.Tensor(arg172_1, 1)
        squeeze_dims_75: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_55, [0, 2, 3]);  getitem_55 = None
        squeeze_dims_76: "f32[120]" = torch.ops.aten.squeeze.dims(rsqrt_27, [0, 2, 3]);  rsqrt_27 = None
        mul_tensor_135: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_dims_75, 0.1)
        mul_tensor_136: "f32[120]" = torch.ops.aten.mul.Tensor(arg173_1, 0.9)
        add_tensor_82: "f32[120]" = torch.ops.aten.add.Tensor(mul_tensor_135, mul_tensor_136);  mul_tensor_135 = mul_tensor_136 = None
        squeeze_dims_77: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_54, [0, 2, 3]);  getitem_54 = None
        mul_tensor_137: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_dims_77, 1.0001594642002871);  squeeze_dims_77 = None
        mul_tensor_138: "f32[120]" = torch.ops.aten.mul.Tensor(mul_tensor_137, 0.1);  mul_tensor_137 = None
        mul_tensor_139: "f32[120]" = torch.ops.aten.mul.Tensor(arg174_1, 0.9)
        add_tensor_83: "f32[120]" = torch.ops.aten.add.Tensor(mul_tensor_138, mul_tensor_139);  mul_tensor_138 = mul_tensor_139 = None
        add_tensor_84: "i64[]" = torch.ops.aten.add.Tensor(arg178_1, 1)
        squeeze_dims_78: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_57, [0, 2, 3]);  getitem_57 = None
        mul_tensor_140: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_dims_78, 0.1);  squeeze_dims_78 = None
        mul_tensor_141: "f32[120]" = torch.ops.aten.mul.Tensor(arg179_1, 0.9)
        add_tensor_85: "f32[120]" = torch.ops.aten.add.Tensor(mul_tensor_140, mul_tensor_141);  mul_tensor_140 = mul_tensor_141 = None
        squeeze_dims_79: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_56, [0, 2, 3]);  getitem_56 = None
        mul_tensor_142: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_dims_79, 1.0001594642002871);  squeeze_dims_79 = None
        mul_tensor_143: "f32[120]" = torch.ops.aten.mul.Tensor(mul_tensor_142, 0.1);  mul_tensor_142 = None
        mul_tensor_144: "f32[120]" = torch.ops.aten.mul.Tensor(arg180_1, 0.9)
        add_tensor_86: "f32[120]" = torch.ops.aten.add.Tensor(mul_tensor_143, mul_tensor_144);  mul_tensor_143 = mul_tensor_144 = None
        add_tensor_87: "i64[]" = torch.ops.aten.add.Tensor(arg184_1, 1)
        squeeze_dims_80: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_59, [0, 2, 3]);  getitem_59 = None
        squeeze_dims_81: "f32[240]" = torch.ops.aten.squeeze.dims(rsqrt_29, [0, 2, 3]);  rsqrt_29 = None
        mul_tensor_145: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_dims_80, 0.1)
        mul_tensor_146: "f32[240]" = torch.ops.aten.mul.Tensor(arg185_1, 0.9)
        add_tensor_88: "f32[240]" = torch.ops.aten.add.Tensor(mul_tensor_145, mul_tensor_146);  mul_tensor_145 = mul_tensor_146 = None
        squeeze_dims_82: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_58, [0, 2, 3]);  getitem_58 = None
        mul_tensor_147: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_dims_82, 1.0006381620931717);  squeeze_dims_82 = None
        mul_tensor_148: "f32[240]" = torch.ops.aten.mul.Tensor(mul_tensor_147, 0.1);  mul_tensor_147 = None
        mul_tensor_149: "f32[240]" = torch.ops.aten.mul.Tensor(arg186_1, 0.9)
        add_tensor_89: "f32[240]" = torch.ops.aten.add.Tensor(mul_tensor_148, mul_tensor_149);  mul_tensor_148 = mul_tensor_149 = None
        add_tensor_90: "i64[]" = torch.ops.aten.add.Tensor(arg190_1, 1)
        squeeze_dims_83: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_61, [0, 2, 3]);  getitem_61 = None
        squeeze_dims_84: "f32[40]" = torch.ops.aten.squeeze.dims(rsqrt_30, [0, 2, 3]);  rsqrt_30 = None
        mul_tensor_150: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_dims_83, 0.1)
        mul_tensor_151: "f32[40]" = torch.ops.aten.mul.Tensor(arg191_1, 0.9)
        add_tensor_91: "f32[40]" = torch.ops.aten.add.Tensor(mul_tensor_150, mul_tensor_151);  mul_tensor_150 = mul_tensor_151 = None
        squeeze_dims_85: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_60, [0, 2, 3]);  getitem_60 = None
        mul_tensor_152: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_dims_85, 1.0006381620931717);  squeeze_dims_85 = None
        mul_tensor_153: "f32[40]" = torch.ops.aten.mul.Tensor(mul_tensor_152, 0.1);  mul_tensor_152 = None
        mul_tensor_154: "f32[40]" = torch.ops.aten.mul.Tensor(arg192_1, 0.9)
        add_tensor_92: "f32[40]" = torch.ops.aten.add.Tensor(mul_tensor_153, mul_tensor_154);  mul_tensor_153 = mul_tensor_154 = None
        add_tensor_93: "i64[]" = torch.ops.aten.add.Tensor(arg196_1, 1)
        squeeze_dims_86: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_63, [0, 2, 3]);  getitem_63 = None
        squeeze_dims_87: "f32[40]" = torch.ops.aten.squeeze.dims(rsqrt_31, [0, 2, 3]);  rsqrt_31 = None
        mul_tensor_155: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_dims_86, 0.1)
        mul_tensor_156: "f32[40]" = torch.ops.aten.mul.Tensor(arg197_1, 0.9)
        add_tensor_94: "f32[40]" = torch.ops.aten.add.Tensor(mul_tensor_155, mul_tensor_156);  mul_tensor_155 = mul_tensor_156 = None
        squeeze_dims_88: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_62, [0, 2, 3]);  getitem_62 = None
        mul_tensor_157: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_dims_88, 1.0006381620931717);  squeeze_dims_88 = None
        mul_tensor_158: "f32[40]" = torch.ops.aten.mul.Tensor(mul_tensor_157, 0.1);  mul_tensor_157 = None
        mul_tensor_159: "f32[40]" = torch.ops.aten.mul.Tensor(arg198_1, 0.9)
        add_tensor_95: "f32[40]" = torch.ops.aten.add.Tensor(mul_tensor_158, mul_tensor_159);  mul_tensor_158 = mul_tensor_159 = None
        add_tensor_96: "i64[]" = torch.ops.aten.add.Tensor(arg202_1, 1)
        squeeze_dims_89: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_65, [0, 2, 3]);  getitem_65 = None
        squeeze_dims_90: "f32[40]" = torch.ops.aten.squeeze.dims(rsqrt_32, [0, 2, 3]);  rsqrt_32 = None
        mul_tensor_160: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_dims_89, 0.1)
        mul_tensor_161: "f32[40]" = torch.ops.aten.mul.Tensor(arg203_1, 0.9)
        add_tensor_97: "f32[40]" = torch.ops.aten.add.Tensor(mul_tensor_160, mul_tensor_161);  mul_tensor_160 = mul_tensor_161 = None
        squeeze_dims_91: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_64, [0, 2, 3]);  getitem_64 = None
        mul_tensor_162: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_dims_91, 1.0006381620931717);  squeeze_dims_91 = None
        mul_tensor_163: "f32[40]" = torch.ops.aten.mul.Tensor(mul_tensor_162, 0.1);  mul_tensor_162 = None
        mul_tensor_164: "f32[40]" = torch.ops.aten.mul.Tensor(arg204_1, 0.9)
        add_tensor_98: "f32[40]" = torch.ops.aten.add.Tensor(mul_tensor_163, mul_tensor_164);  mul_tensor_163 = mul_tensor_164 = None
        add_tensor_99: "i64[]" = torch.ops.aten.add.Tensor(arg208_1, 1)
        squeeze_dims_92: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_67, [0, 2, 3]);  getitem_67 = None
        squeeze_dims_93: "f32[80]" = torch.ops.aten.squeeze.dims(rsqrt_33, [0, 2, 3]);  rsqrt_33 = None
        mul_tensor_165: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_dims_92, 0.1)
        mul_tensor_166: "f32[80]" = torch.ops.aten.mul.Tensor(arg209_1, 0.9)
        add_tensor_100: "f32[80]" = torch.ops.aten.add.Tensor(mul_tensor_165, mul_tensor_166);  mul_tensor_165 = mul_tensor_166 = None
        squeeze_dims_94: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_66, [0, 2, 3]);  getitem_66 = None
        mul_tensor_167: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_dims_94, 1.0006381620931717);  squeeze_dims_94 = None
        mul_tensor_168: "f32[80]" = torch.ops.aten.mul.Tensor(mul_tensor_167, 0.1);  mul_tensor_167 = None
        mul_tensor_169: "f32[80]" = torch.ops.aten.mul.Tensor(arg210_1, 0.9)
        add_tensor_101: "f32[80]" = torch.ops.aten.add.Tensor(mul_tensor_168, mul_tensor_169);  mul_tensor_168 = mul_tensor_169 = None
        add_tensor_102: "i64[]" = torch.ops.aten.add.Tensor(arg214_1, 1)
        squeeze_dims_95: "f32[100]" = torch.ops.aten.squeeze.dims(getitem_69, [0, 2, 3]);  getitem_69 = None
        squeeze_dims_96: "f32[100]" = torch.ops.aten.squeeze.dims(rsqrt_34, [0, 2, 3]);  rsqrt_34 = None
        mul_tensor_170: "f32[100]" = torch.ops.aten.mul.Tensor(squeeze_dims_95, 0.1)
        mul_tensor_171: "f32[100]" = torch.ops.aten.mul.Tensor(arg215_1, 0.9)
        add_tensor_103: "f32[100]" = torch.ops.aten.add.Tensor(mul_tensor_170, mul_tensor_171);  mul_tensor_170 = mul_tensor_171 = None
        squeeze_dims_97: "f32[100]" = torch.ops.aten.squeeze.dims(getitem_68, [0, 2, 3]);  getitem_68 = None
        mul_tensor_172: "f32[100]" = torch.ops.aten.mul.Tensor(squeeze_dims_97, 1.0006381620931717);  squeeze_dims_97 = None
        mul_tensor_173: "f32[100]" = torch.ops.aten.mul.Tensor(mul_tensor_172, 0.1);  mul_tensor_172 = None
        mul_tensor_174: "f32[100]" = torch.ops.aten.mul.Tensor(arg216_1, 0.9)
        add_tensor_104: "f32[100]" = torch.ops.aten.add.Tensor(mul_tensor_173, mul_tensor_174);  mul_tensor_173 = mul_tensor_174 = None
        add_tensor_105: "i64[]" = torch.ops.aten.add.Tensor(arg220_1, 1)
        squeeze_dims_98: "f32[100]" = torch.ops.aten.squeeze.dims(getitem_71, [0, 2, 3]);  getitem_71 = None
        mul_tensor_175: "f32[100]" = torch.ops.aten.mul.Tensor(squeeze_dims_98, 0.1);  squeeze_dims_98 = None
        mul_tensor_176: "f32[100]" = torch.ops.aten.mul.Tensor(arg221_1, 0.9)
        add_tensor_106: "f32[100]" = torch.ops.aten.add.Tensor(mul_tensor_175, mul_tensor_176);  mul_tensor_175 = mul_tensor_176 = None
        squeeze_dims_99: "f32[100]" = torch.ops.aten.squeeze.dims(getitem_70, [0, 2, 3]);  getitem_70 = None
        mul_tensor_177: "f32[100]" = torch.ops.aten.mul.Tensor(squeeze_dims_99, 1.0006381620931717);  squeeze_dims_99 = None
        mul_tensor_178: "f32[100]" = torch.ops.aten.mul.Tensor(mul_tensor_177, 0.1);  mul_tensor_177 = None
        mul_tensor_179: "f32[100]" = torch.ops.aten.mul.Tensor(arg222_1, 0.9)
        add_tensor_107: "f32[100]" = torch.ops.aten.add.Tensor(mul_tensor_178, mul_tensor_179);  mul_tensor_178 = mul_tensor_179 = None
        add_tensor_108: "i64[]" = torch.ops.aten.add.Tensor(arg226_1, 1)
        squeeze_dims_100: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_73, [0, 2, 3]);  getitem_73 = None
        squeeze_dims_101: "f32[40]" = torch.ops.aten.squeeze.dims(rsqrt_36, [0, 2, 3]);  rsqrt_36 = None
        mul_tensor_180: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_dims_100, 0.1)
        mul_tensor_181: "f32[40]" = torch.ops.aten.mul.Tensor(arg227_1, 0.9)
        add_tensor_109: "f32[40]" = torch.ops.aten.add.Tensor(mul_tensor_180, mul_tensor_181);  mul_tensor_180 = mul_tensor_181 = None
        squeeze_dims_102: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_72, [0, 2, 3]);  getitem_72 = None
        mul_tensor_182: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_dims_102, 1.0006381620931717);  squeeze_dims_102 = None
        mul_tensor_183: "f32[40]" = torch.ops.aten.mul.Tensor(mul_tensor_182, 0.1);  mul_tensor_182 = None
        mul_tensor_184: "f32[40]" = torch.ops.aten.mul.Tensor(arg228_1, 0.9)
        add_tensor_110: "f32[40]" = torch.ops.aten.add.Tensor(mul_tensor_183, mul_tensor_184);  mul_tensor_183 = mul_tensor_184 = None
        add_tensor_111: "i64[]" = torch.ops.aten.add.Tensor(arg232_1, 1)
        squeeze_dims_103: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_75, [0, 2, 3]);  getitem_75 = None
        squeeze_dims_104: "f32[40]" = torch.ops.aten.squeeze.dims(rsqrt_37, [0, 2, 3]);  rsqrt_37 = None
        mul_tensor_185: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_dims_103, 0.1)
        mul_tensor_186: "f32[40]" = torch.ops.aten.mul.Tensor(arg233_1, 0.9)
        add_tensor_112: "f32[40]" = torch.ops.aten.add.Tensor(mul_tensor_185, mul_tensor_186);  mul_tensor_185 = mul_tensor_186 = None
        squeeze_dims_105: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_74, [0, 2, 3]);  getitem_74 = None
        mul_tensor_187: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_dims_105, 1.0006381620931717);  squeeze_dims_105 = None
        mul_tensor_188: "f32[40]" = torch.ops.aten.mul.Tensor(mul_tensor_187, 0.1);  mul_tensor_187 = None
        mul_tensor_189: "f32[40]" = torch.ops.aten.mul.Tensor(arg234_1, 0.9)
        add_tensor_113: "f32[40]" = torch.ops.aten.add.Tensor(mul_tensor_188, mul_tensor_189);  mul_tensor_188 = mul_tensor_189 = None
        add_tensor_114: "i64[]" = torch.ops.aten.add.Tensor(arg238_1, 1)
        squeeze_dims_106: "f32[92]" = torch.ops.aten.squeeze.dims(getitem_77, [0, 2, 3]);  getitem_77 = None
        squeeze_dims_107: "f32[92]" = torch.ops.aten.squeeze.dims(rsqrt_38, [0, 2, 3]);  rsqrt_38 = None
        mul_tensor_190: "f32[92]" = torch.ops.aten.mul.Tensor(squeeze_dims_106, 0.1)
        mul_tensor_191: "f32[92]" = torch.ops.aten.mul.Tensor(arg239_1, 0.9)
        add_tensor_115: "f32[92]" = torch.ops.aten.add.Tensor(mul_tensor_190, mul_tensor_191);  mul_tensor_190 = mul_tensor_191 = None
        squeeze_dims_108: "f32[92]" = torch.ops.aten.squeeze.dims(getitem_76, [0, 2, 3]);  getitem_76 = None
        mul_tensor_192: "f32[92]" = torch.ops.aten.mul.Tensor(squeeze_dims_108, 1.0006381620931717);  squeeze_dims_108 = None
        mul_tensor_193: "f32[92]" = torch.ops.aten.mul.Tensor(mul_tensor_192, 0.1);  mul_tensor_192 = None
        mul_tensor_194: "f32[92]" = torch.ops.aten.mul.Tensor(arg240_1, 0.9)
        add_tensor_116: "f32[92]" = torch.ops.aten.add.Tensor(mul_tensor_193, mul_tensor_194);  mul_tensor_193 = mul_tensor_194 = None
        add_tensor_117: "i64[]" = torch.ops.aten.add.Tensor(arg244_1, 1)
        squeeze_dims_109: "f32[92]" = torch.ops.aten.squeeze.dims(getitem_79, [0, 2, 3]);  getitem_79 = None
        mul_tensor_195: "f32[92]" = torch.ops.aten.mul.Tensor(squeeze_dims_109, 0.1);  squeeze_dims_109 = None
        mul_tensor_196: "f32[92]" = torch.ops.aten.mul.Tensor(arg245_1, 0.9)
        add_tensor_118: "f32[92]" = torch.ops.aten.add.Tensor(mul_tensor_195, mul_tensor_196);  mul_tensor_195 = mul_tensor_196 = None
        squeeze_dims_110: "f32[92]" = torch.ops.aten.squeeze.dims(getitem_78, [0, 2, 3]);  getitem_78 = None
        mul_tensor_197: "f32[92]" = torch.ops.aten.mul.Tensor(squeeze_dims_110, 1.0006381620931717);  squeeze_dims_110 = None
        mul_tensor_198: "f32[92]" = torch.ops.aten.mul.Tensor(mul_tensor_197, 0.1);  mul_tensor_197 = None
        mul_tensor_199: "f32[92]" = torch.ops.aten.mul.Tensor(arg246_1, 0.9)
        add_tensor_119: "f32[92]" = torch.ops.aten.add.Tensor(mul_tensor_198, mul_tensor_199);  mul_tensor_198 = mul_tensor_199 = None
        add_tensor_120: "i64[]" = torch.ops.aten.add.Tensor(arg250_1, 1)
        squeeze_dims_111: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_81, [0, 2, 3]);  getitem_81 = None
        squeeze_dims_112: "f32[40]" = torch.ops.aten.squeeze.dims(rsqrt_40, [0, 2, 3]);  rsqrt_40 = None
        mul_tensor_200: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_dims_111, 0.1)
        mul_tensor_201: "f32[40]" = torch.ops.aten.mul.Tensor(arg251_1, 0.9)
        add_tensor_121: "f32[40]" = torch.ops.aten.add.Tensor(mul_tensor_200, mul_tensor_201);  mul_tensor_200 = mul_tensor_201 = None
        squeeze_dims_113: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_80, [0, 2, 3]);  getitem_80 = None
        mul_tensor_202: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_dims_113, 1.0006381620931717);  squeeze_dims_113 = None
        mul_tensor_203: "f32[40]" = torch.ops.aten.mul.Tensor(mul_tensor_202, 0.1);  mul_tensor_202 = None
        mul_tensor_204: "f32[40]" = torch.ops.aten.mul.Tensor(arg252_1, 0.9)
        add_tensor_122: "f32[40]" = torch.ops.aten.add.Tensor(mul_tensor_203, mul_tensor_204);  mul_tensor_203 = mul_tensor_204 = None
        add_tensor_123: "i64[]" = torch.ops.aten.add.Tensor(arg256_1, 1)
        squeeze_dims_114: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_83, [0, 2, 3]);  getitem_83 = None
        squeeze_dims_115: "f32[40]" = torch.ops.aten.squeeze.dims(rsqrt_41, [0, 2, 3]);  rsqrt_41 = None
        mul_tensor_205: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_dims_114, 0.1)
        mul_tensor_206: "f32[40]" = torch.ops.aten.mul.Tensor(arg257_1, 0.9)
        add_tensor_124: "f32[40]" = torch.ops.aten.add.Tensor(mul_tensor_205, mul_tensor_206);  mul_tensor_205 = mul_tensor_206 = None
        squeeze_dims_116: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_82, [0, 2, 3]);  getitem_82 = None
        mul_tensor_207: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_dims_116, 1.0006381620931717);  squeeze_dims_116 = None
        mul_tensor_208: "f32[40]" = torch.ops.aten.mul.Tensor(mul_tensor_207, 0.1);  mul_tensor_207 = None
        mul_tensor_209: "f32[40]" = torch.ops.aten.mul.Tensor(arg258_1, 0.9)
        add_tensor_125: "f32[40]" = torch.ops.aten.add.Tensor(mul_tensor_208, mul_tensor_209);  mul_tensor_208 = mul_tensor_209 = None
        add_tensor_126: "i64[]" = torch.ops.aten.add.Tensor(arg262_1, 1)
        squeeze_dims_117: "f32[92]" = torch.ops.aten.squeeze.dims(getitem_85, [0, 2, 3]);  getitem_85 = None
        squeeze_dims_118: "f32[92]" = torch.ops.aten.squeeze.dims(rsqrt_42, [0, 2, 3]);  rsqrt_42 = None
        mul_tensor_210: "f32[92]" = torch.ops.aten.mul.Tensor(squeeze_dims_117, 0.1)
        mul_tensor_211: "f32[92]" = torch.ops.aten.mul.Tensor(arg263_1, 0.9)
        add_tensor_127: "f32[92]" = torch.ops.aten.add.Tensor(mul_tensor_210, mul_tensor_211);  mul_tensor_210 = mul_tensor_211 = None
        squeeze_dims_119: "f32[92]" = torch.ops.aten.squeeze.dims(getitem_84, [0, 2, 3]);  getitem_84 = None
        mul_tensor_212: "f32[92]" = torch.ops.aten.mul.Tensor(squeeze_dims_119, 1.0006381620931717);  squeeze_dims_119 = None
        mul_tensor_213: "f32[92]" = torch.ops.aten.mul.Tensor(mul_tensor_212, 0.1);  mul_tensor_212 = None
        mul_tensor_214: "f32[92]" = torch.ops.aten.mul.Tensor(arg264_1, 0.9)
        add_tensor_128: "f32[92]" = torch.ops.aten.add.Tensor(mul_tensor_213, mul_tensor_214);  mul_tensor_213 = mul_tensor_214 = None
        add_tensor_129: "i64[]" = torch.ops.aten.add.Tensor(arg268_1, 1)
        squeeze_dims_120: "f32[92]" = torch.ops.aten.squeeze.dims(getitem_87, [0, 2, 3]);  getitem_87 = None
        mul_tensor_215: "f32[92]" = torch.ops.aten.mul.Tensor(squeeze_dims_120, 0.1);  squeeze_dims_120 = None
        mul_tensor_216: "f32[92]" = torch.ops.aten.mul.Tensor(arg269_1, 0.9)
        add_tensor_130: "f32[92]" = torch.ops.aten.add.Tensor(mul_tensor_215, mul_tensor_216);  mul_tensor_215 = mul_tensor_216 = None
        squeeze_dims_121: "f32[92]" = torch.ops.aten.squeeze.dims(getitem_86, [0, 2, 3]);  getitem_86 = None
        mul_tensor_217: "f32[92]" = torch.ops.aten.mul.Tensor(squeeze_dims_121, 1.0006381620931717);  squeeze_dims_121 = None
        mul_tensor_218: "f32[92]" = torch.ops.aten.mul.Tensor(mul_tensor_217, 0.1);  mul_tensor_217 = None
        mul_tensor_219: "f32[92]" = torch.ops.aten.mul.Tensor(arg270_1, 0.9)
        add_tensor_131: "f32[92]" = torch.ops.aten.add.Tensor(mul_tensor_218, mul_tensor_219);  mul_tensor_218 = mul_tensor_219 = None
        add_tensor_132: "i64[]" = torch.ops.aten.add.Tensor(arg274_1, 1)
        squeeze_dims_122: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_89, [0, 2, 3]);  getitem_89 = None
        squeeze_dims_123: "f32[40]" = torch.ops.aten.squeeze.dims(rsqrt_44, [0, 2, 3]);  rsqrt_44 = None
        mul_tensor_220: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_dims_122, 0.1)
        mul_tensor_221: "f32[40]" = torch.ops.aten.mul.Tensor(arg275_1, 0.9)
        add_tensor_133: "f32[40]" = torch.ops.aten.add.Tensor(mul_tensor_220, mul_tensor_221);  mul_tensor_220 = mul_tensor_221 = None
        squeeze_dims_124: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_88, [0, 2, 3]);  getitem_88 = None
        mul_tensor_222: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_dims_124, 1.0006381620931717);  squeeze_dims_124 = None
        mul_tensor_223: "f32[40]" = torch.ops.aten.mul.Tensor(mul_tensor_222, 0.1);  mul_tensor_222 = None
        mul_tensor_224: "f32[40]" = torch.ops.aten.mul.Tensor(arg276_1, 0.9)
        add_tensor_134: "f32[40]" = torch.ops.aten.add.Tensor(mul_tensor_223, mul_tensor_224);  mul_tensor_223 = mul_tensor_224 = None
        add_tensor_135: "i64[]" = torch.ops.aten.add.Tensor(arg280_1, 1)
        squeeze_dims_125: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_91, [0, 2, 3]);  getitem_91 = None
        squeeze_dims_126: "f32[40]" = torch.ops.aten.squeeze.dims(rsqrt_45, [0, 2, 3]);  rsqrt_45 = None
        mul_tensor_225: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_dims_125, 0.1)
        mul_tensor_226: "f32[40]" = torch.ops.aten.mul.Tensor(arg281_1, 0.9)
        add_tensor_136: "f32[40]" = torch.ops.aten.add.Tensor(mul_tensor_225, mul_tensor_226);  mul_tensor_225 = mul_tensor_226 = None
        squeeze_dims_127: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_90, [0, 2, 3]);  getitem_90 = None
        mul_tensor_227: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_dims_127, 1.0006381620931717);  squeeze_dims_127 = None
        mul_tensor_228: "f32[40]" = torch.ops.aten.mul.Tensor(mul_tensor_227, 0.1);  mul_tensor_227 = None
        mul_tensor_229: "f32[40]" = torch.ops.aten.mul.Tensor(arg282_1, 0.9)
        add_tensor_137: "f32[40]" = torch.ops.aten.add.Tensor(mul_tensor_228, mul_tensor_229);  mul_tensor_228 = mul_tensor_229 = None
        add_tensor_138: "i64[]" = torch.ops.aten.add.Tensor(arg286_1, 1)
        squeeze_dims_128: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_93, [0, 2, 3]);  getitem_93 = None
        squeeze_dims_129: "f32[240]" = torch.ops.aten.squeeze.dims(rsqrt_46, [0, 2, 3]);  rsqrt_46 = None
        mul_tensor_230: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_dims_128, 0.1)
        mul_tensor_231: "f32[240]" = torch.ops.aten.mul.Tensor(arg287_1, 0.9)
        add_tensor_139: "f32[240]" = torch.ops.aten.add.Tensor(mul_tensor_230, mul_tensor_231);  mul_tensor_230 = mul_tensor_231 = None
        squeeze_dims_130: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_92, [0, 2, 3]);  getitem_92 = None
        mul_tensor_232: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_dims_130, 1.0006381620931717);  squeeze_dims_130 = None
        mul_tensor_233: "f32[240]" = torch.ops.aten.mul.Tensor(mul_tensor_232, 0.1);  mul_tensor_232 = None
        mul_tensor_234: "f32[240]" = torch.ops.aten.mul.Tensor(arg288_1, 0.9)
        add_tensor_140: "f32[240]" = torch.ops.aten.add.Tensor(mul_tensor_233, mul_tensor_234);  mul_tensor_233 = mul_tensor_234 = None
        add_tensor_141: "i64[]" = torch.ops.aten.add.Tensor(arg292_1, 1)
        squeeze_dims_131: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_95, [0, 2, 3]);  getitem_95 = None
        mul_tensor_235: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_dims_131, 0.1);  squeeze_dims_131 = None
        mul_tensor_236: "f32[240]" = torch.ops.aten.mul.Tensor(arg293_1, 0.9)
        add_tensor_142: "f32[240]" = torch.ops.aten.add.Tensor(mul_tensor_235, mul_tensor_236);  mul_tensor_235 = mul_tensor_236 = None
        squeeze_dims_132: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_94, [0, 2, 3]);  getitem_94 = None
        mul_tensor_237: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_dims_132, 1.0006381620931717);  squeeze_dims_132 = None
        mul_tensor_238: "f32[240]" = torch.ops.aten.mul.Tensor(mul_tensor_237, 0.1);  mul_tensor_237 = None
        mul_tensor_239: "f32[240]" = torch.ops.aten.mul.Tensor(arg294_1, 0.9)
        add_tensor_143: "f32[240]" = torch.ops.aten.add.Tensor(mul_tensor_238, mul_tensor_239);  mul_tensor_238 = mul_tensor_239 = None
        add_tensor_144: "i64[]" = torch.ops.aten.add.Tensor(arg302_1, 1)
        squeeze_dims_133: "f32[56]" = torch.ops.aten.squeeze.dims(getitem_97, [0, 2, 3]);  getitem_97 = None
        squeeze_dims_134: "f32[56]" = torch.ops.aten.squeeze.dims(rsqrt_48, [0, 2, 3]);  rsqrt_48 = None
        mul_tensor_240: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_dims_133, 0.1)
        mul_tensor_241: "f32[56]" = torch.ops.aten.mul.Tensor(arg303_1, 0.9)
        add_tensor_145: "f32[56]" = torch.ops.aten.add.Tensor(mul_tensor_240, mul_tensor_241);  mul_tensor_240 = mul_tensor_241 = None
        squeeze_dims_135: "f32[56]" = torch.ops.aten.squeeze.dims(getitem_96, [0, 2, 3]);  getitem_96 = None
        mul_tensor_242: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_dims_135, 1.0006381620931717);  squeeze_dims_135 = None
        mul_tensor_243: "f32[56]" = torch.ops.aten.mul.Tensor(mul_tensor_242, 0.1);  mul_tensor_242 = None
        mul_tensor_244: "f32[56]" = torch.ops.aten.mul.Tensor(arg304_1, 0.9)
        add_tensor_146: "f32[56]" = torch.ops.aten.add.Tensor(mul_tensor_243, mul_tensor_244);  mul_tensor_243 = mul_tensor_244 = None
        add_tensor_147: "i64[]" = torch.ops.aten.add.Tensor(arg308_1, 1)
        squeeze_dims_136: "f32[56]" = torch.ops.aten.squeeze.dims(getitem_99, [0, 2, 3]);  getitem_99 = None
        squeeze_dims_137: "f32[56]" = torch.ops.aten.squeeze.dims(rsqrt_49, [0, 2, 3]);  rsqrt_49 = None
        mul_tensor_245: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_dims_136, 0.1)
        mul_tensor_246: "f32[56]" = torch.ops.aten.mul.Tensor(arg309_1, 0.9)
        add_tensor_148: "f32[56]" = torch.ops.aten.add.Tensor(mul_tensor_245, mul_tensor_246);  mul_tensor_245 = mul_tensor_246 = None
        squeeze_dims_138: "f32[56]" = torch.ops.aten.squeeze.dims(getitem_98, [0, 2, 3]);  getitem_98 = None
        mul_tensor_247: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_dims_138, 1.0006381620931717);  squeeze_dims_138 = None
        mul_tensor_248: "f32[56]" = torch.ops.aten.mul.Tensor(mul_tensor_247, 0.1);  mul_tensor_247 = None
        mul_tensor_249: "f32[56]" = torch.ops.aten.mul.Tensor(arg310_1, 0.9)
        add_tensor_149: "f32[56]" = torch.ops.aten.add.Tensor(mul_tensor_248, mul_tensor_249);  mul_tensor_248 = mul_tensor_249 = None
        add_tensor_150: "i64[]" = torch.ops.aten.add.Tensor(arg314_1, 1)
        squeeze_dims_139: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_101, [0, 2, 3]);  getitem_101 = None
        squeeze_dims_140: "f32[80]" = torch.ops.aten.squeeze.dims(rsqrt_50, [0, 2, 3]);  rsqrt_50 = None
        mul_tensor_250: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_dims_139, 0.1)
        mul_tensor_251: "f32[80]" = torch.ops.aten.mul.Tensor(arg315_1, 0.9)
        add_tensor_151: "f32[80]" = torch.ops.aten.add.Tensor(mul_tensor_250, mul_tensor_251);  mul_tensor_250 = mul_tensor_251 = None
        squeeze_dims_141: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_100, [0, 2, 3]);  getitem_100 = None
        mul_tensor_252: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_dims_141, 1.0006381620931717);  squeeze_dims_141 = None
        mul_tensor_253: "f32[80]" = torch.ops.aten.mul.Tensor(mul_tensor_252, 0.1);  mul_tensor_252 = None
        mul_tensor_254: "f32[80]" = torch.ops.aten.mul.Tensor(arg316_1, 0.9)
        add_tensor_152: "f32[80]" = torch.ops.aten.add.Tensor(mul_tensor_253, mul_tensor_254);  mul_tensor_253 = mul_tensor_254 = None
        add_tensor_153: "i64[]" = torch.ops.aten.add.Tensor(arg320_1, 1)
        squeeze_dims_142: "f32[112]" = torch.ops.aten.squeeze.dims(getitem_103, [0, 2, 3]);  getitem_103 = None
        squeeze_dims_143: "f32[112]" = torch.ops.aten.squeeze.dims(rsqrt_51, [0, 2, 3]);  rsqrt_51 = None
        mul_tensor_255: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_dims_142, 0.1)
        mul_tensor_256: "f32[112]" = torch.ops.aten.mul.Tensor(arg321_1, 0.9)
        add_tensor_154: "f32[112]" = torch.ops.aten.add.Tensor(mul_tensor_255, mul_tensor_256);  mul_tensor_255 = mul_tensor_256 = None
        squeeze_dims_144: "f32[112]" = torch.ops.aten.squeeze.dims(getitem_102, [0, 2, 3]);  getitem_102 = None
        mul_tensor_257: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_dims_144, 1.0006381620931717);  squeeze_dims_144 = None
        mul_tensor_258: "f32[112]" = torch.ops.aten.mul.Tensor(mul_tensor_257, 0.1);  mul_tensor_257 = None
        mul_tensor_259: "f32[112]" = torch.ops.aten.mul.Tensor(arg322_1, 0.9)
        add_tensor_155: "f32[112]" = torch.ops.aten.add.Tensor(mul_tensor_258, mul_tensor_259);  mul_tensor_258 = mul_tensor_259 = None
        add_tensor_156: "i64[]" = torch.ops.aten.add.Tensor(arg326_1, 1)
        squeeze_dims_145: "f32[336]" = torch.ops.aten.squeeze.dims(getitem_105, [0, 2, 3]);  getitem_105 = None
        squeeze_dims_146: "f32[336]" = torch.ops.aten.squeeze.dims(rsqrt_52, [0, 2, 3]);  rsqrt_52 = None
        mul_tensor_260: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_dims_145, 0.1)
        mul_tensor_261: "f32[336]" = torch.ops.aten.mul.Tensor(arg327_1, 0.9)
        add_tensor_157: "f32[336]" = torch.ops.aten.add.Tensor(mul_tensor_260, mul_tensor_261);  mul_tensor_260 = mul_tensor_261 = None
        squeeze_dims_147: "f32[336]" = torch.ops.aten.squeeze.dims(getitem_104, [0, 2, 3]);  getitem_104 = None
        mul_tensor_262: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_dims_147, 1.0006381620931717);  squeeze_dims_147 = None
        mul_tensor_263: "f32[336]" = torch.ops.aten.mul.Tensor(mul_tensor_262, 0.1);  mul_tensor_262 = None
        mul_tensor_264: "f32[336]" = torch.ops.aten.mul.Tensor(arg328_1, 0.9)
        add_tensor_158: "f32[336]" = torch.ops.aten.add.Tensor(mul_tensor_263, mul_tensor_264);  mul_tensor_263 = mul_tensor_264 = None
        add_tensor_159: "i64[]" = torch.ops.aten.add.Tensor(arg332_1, 1)
        squeeze_dims_148: "f32[336]" = torch.ops.aten.squeeze.dims(getitem_107, [0, 2, 3]);  getitem_107 = None
        mul_tensor_265: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_dims_148, 0.1);  squeeze_dims_148 = None
        mul_tensor_266: "f32[336]" = torch.ops.aten.mul.Tensor(arg333_1, 0.9)
        add_tensor_160: "f32[336]" = torch.ops.aten.add.Tensor(mul_tensor_265, mul_tensor_266);  mul_tensor_265 = mul_tensor_266 = None
        squeeze_dims_149: "f32[336]" = torch.ops.aten.squeeze.dims(getitem_106, [0, 2, 3]);  getitem_106 = None
        mul_tensor_267: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_dims_149, 1.0006381620931717);  squeeze_dims_149 = None
        mul_tensor_268: "f32[336]" = torch.ops.aten.mul.Tensor(mul_tensor_267, 0.1);  mul_tensor_267 = None
        mul_tensor_269: "f32[336]" = torch.ops.aten.mul.Tensor(arg334_1, 0.9)
        add_tensor_161: "f32[336]" = torch.ops.aten.add.Tensor(mul_tensor_268, mul_tensor_269);  mul_tensor_268 = mul_tensor_269 = None
        add_tensor_162: "i64[]" = torch.ops.aten.add.Tensor(arg342_1, 1)
        squeeze_dims_150: "f32[56]" = torch.ops.aten.squeeze.dims(getitem_109, [0, 2, 3]);  getitem_109 = None
        squeeze_dims_151: "f32[56]" = torch.ops.aten.squeeze.dims(rsqrt_54, [0, 2, 3]);  rsqrt_54 = None
        mul_tensor_270: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_dims_150, 0.1)
        mul_tensor_271: "f32[56]" = torch.ops.aten.mul.Tensor(arg343_1, 0.9)
        add_tensor_163: "f32[56]" = torch.ops.aten.add.Tensor(mul_tensor_270, mul_tensor_271);  mul_tensor_270 = mul_tensor_271 = None
        squeeze_dims_152: "f32[56]" = torch.ops.aten.squeeze.dims(getitem_108, [0, 2, 3]);  getitem_108 = None
        mul_tensor_272: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_dims_152, 1.0006381620931717);  squeeze_dims_152 = None
        mul_tensor_273: "f32[56]" = torch.ops.aten.mul.Tensor(mul_tensor_272, 0.1);  mul_tensor_272 = None
        mul_tensor_274: "f32[56]" = torch.ops.aten.mul.Tensor(arg344_1, 0.9)
        add_tensor_164: "f32[56]" = torch.ops.aten.add.Tensor(mul_tensor_273, mul_tensor_274);  mul_tensor_273 = mul_tensor_274 = None
        add_tensor_165: "i64[]" = torch.ops.aten.add.Tensor(arg348_1, 1)
        squeeze_dims_153: "f32[56]" = torch.ops.aten.squeeze.dims(getitem_111, [0, 2, 3]);  getitem_111 = None
        squeeze_dims_154: "f32[56]" = torch.ops.aten.squeeze.dims(rsqrt_55, [0, 2, 3]);  rsqrt_55 = None
        mul_tensor_275: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_dims_153, 0.1)
        mul_tensor_276: "f32[56]" = torch.ops.aten.mul.Tensor(arg349_1, 0.9)
        add_tensor_166: "f32[56]" = torch.ops.aten.add.Tensor(mul_tensor_275, mul_tensor_276);  mul_tensor_275 = mul_tensor_276 = None
        squeeze_dims_155: "f32[56]" = torch.ops.aten.squeeze.dims(getitem_110, [0, 2, 3]);  getitem_110 = None
        mul_tensor_277: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_dims_155, 1.0006381620931717);  squeeze_dims_155 = None
        mul_tensor_278: "f32[56]" = torch.ops.aten.mul.Tensor(mul_tensor_277, 0.1);  mul_tensor_277 = None
        mul_tensor_279: "f32[56]" = torch.ops.aten.mul.Tensor(arg350_1, 0.9)
        add_tensor_167: "f32[56]" = torch.ops.aten.add.Tensor(mul_tensor_278, mul_tensor_279);  mul_tensor_278 = mul_tensor_279 = None
        add_tensor_168: "i64[]" = torch.ops.aten.add.Tensor(arg354_1, 1)
        squeeze_dims_156: "f32[336]" = torch.ops.aten.squeeze.dims(getitem_113, [0, 2, 3]);  getitem_113 = None
        squeeze_dims_157: "f32[336]" = torch.ops.aten.squeeze.dims(rsqrt_56, [0, 2, 3]);  rsqrt_56 = None
        mul_tensor_280: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_dims_156, 0.1)
        mul_tensor_281: "f32[336]" = torch.ops.aten.mul.Tensor(arg355_1, 0.9)
        add_tensor_169: "f32[336]" = torch.ops.aten.add.Tensor(mul_tensor_280, mul_tensor_281);  mul_tensor_280 = mul_tensor_281 = None
        squeeze_dims_158: "f32[336]" = torch.ops.aten.squeeze.dims(getitem_112, [0, 2, 3]);  getitem_112 = None
        mul_tensor_282: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_dims_158, 1.0006381620931717);  squeeze_dims_158 = None
        mul_tensor_283: "f32[336]" = torch.ops.aten.mul.Tensor(mul_tensor_282, 0.1);  mul_tensor_282 = None
        mul_tensor_284: "f32[336]" = torch.ops.aten.mul.Tensor(arg356_1, 0.9)
        add_tensor_170: "f32[336]" = torch.ops.aten.add.Tensor(mul_tensor_283, mul_tensor_284);  mul_tensor_283 = mul_tensor_284 = None
        add_tensor_171: "i64[]" = torch.ops.aten.add.Tensor(arg360_1, 1)
        squeeze_dims_159: "f32[336]" = torch.ops.aten.squeeze.dims(getitem_115, [0, 2, 3]);  getitem_115 = None
        mul_tensor_285: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_dims_159, 0.1);  squeeze_dims_159 = None
        mul_tensor_286: "f32[336]" = torch.ops.aten.mul.Tensor(arg361_1, 0.9)
        add_tensor_172: "f32[336]" = torch.ops.aten.add.Tensor(mul_tensor_285, mul_tensor_286);  mul_tensor_285 = mul_tensor_286 = None
        squeeze_dims_160: "f32[336]" = torch.ops.aten.squeeze.dims(getitem_114, [0, 2, 3]);  getitem_114 = None
        mul_tensor_287: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_dims_160, 1.0006381620931717);  squeeze_dims_160 = None
        mul_tensor_288: "f32[336]" = torch.ops.aten.mul.Tensor(mul_tensor_287, 0.1);  mul_tensor_287 = None
        mul_tensor_289: "f32[336]" = torch.ops.aten.mul.Tensor(arg362_1, 0.9)
        add_tensor_173: "f32[336]" = torch.ops.aten.add.Tensor(mul_tensor_288, mul_tensor_289);  mul_tensor_288 = mul_tensor_289 = None
        add_tensor_174: "i64[]" = torch.ops.aten.add.Tensor(arg366_1, 1)
        squeeze_dims_161: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_117, [0, 2, 3]);  getitem_117 = None
        mul_tensor_290: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_dims_161, 0.1);  squeeze_dims_161 = None
        mul_tensor_291: "f32[672]" = torch.ops.aten.mul.Tensor(arg367_1, 0.9)
        add_tensor_175: "f32[672]" = torch.ops.aten.add.Tensor(mul_tensor_290, mul_tensor_291);  mul_tensor_290 = mul_tensor_291 = None
        squeeze_dims_162: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_116, [0, 2, 3]);  getitem_116 = None
        mul_tensor_292: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_dims_162, 1.0025575447570332);  squeeze_dims_162 = None
        mul_tensor_293: "f32[672]" = torch.ops.aten.mul.Tensor(mul_tensor_292, 0.1);  mul_tensor_292 = None
        mul_tensor_294: "f32[672]" = torch.ops.aten.mul.Tensor(arg368_1, 0.9)
        add_tensor_176: "f32[672]" = torch.ops.aten.add.Tensor(mul_tensor_293, mul_tensor_294);  mul_tensor_293 = mul_tensor_294 = None
        add_tensor_177: "i64[]" = torch.ops.aten.add.Tensor(arg376_1, 1)
        squeeze_dims_163: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_119, [0, 2, 3]);  getitem_119 = None
        squeeze_dims_164: "f32[80]" = torch.ops.aten.squeeze.dims(rsqrt_59, [0, 2, 3]);  rsqrt_59 = None
        mul_tensor_295: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_dims_163, 0.1)
        mul_tensor_296: "f32[80]" = torch.ops.aten.mul.Tensor(arg377_1, 0.9)
        add_tensor_178: "f32[80]" = torch.ops.aten.add.Tensor(mul_tensor_295, mul_tensor_296);  mul_tensor_295 = mul_tensor_296 = None
        squeeze_dims_165: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_118, [0, 2, 3]);  getitem_118 = None
        mul_tensor_297: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_dims_165, 1.0025575447570332);  squeeze_dims_165 = None
        mul_tensor_298: "f32[80]" = torch.ops.aten.mul.Tensor(mul_tensor_297, 0.1);  mul_tensor_297 = None
        mul_tensor_299: "f32[80]" = torch.ops.aten.mul.Tensor(arg378_1, 0.9)
        add_tensor_179: "f32[80]" = torch.ops.aten.add.Tensor(mul_tensor_298, mul_tensor_299);  mul_tensor_298 = mul_tensor_299 = None
        add_tensor_180: "i64[]" = torch.ops.aten.add.Tensor(arg382_1, 1)
        squeeze_dims_166: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_121, [0, 2, 3]);  getitem_121 = None
        squeeze_dims_167: "f32[80]" = torch.ops.aten.squeeze.dims(rsqrt_60, [0, 2, 3]);  rsqrt_60 = None
        mul_tensor_300: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_dims_166, 0.1)
        mul_tensor_301: "f32[80]" = torch.ops.aten.mul.Tensor(arg383_1, 0.9)
        add_tensor_181: "f32[80]" = torch.ops.aten.add.Tensor(mul_tensor_300, mul_tensor_301);  mul_tensor_300 = mul_tensor_301 = None
        squeeze_dims_168: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_120, [0, 2, 3]);  getitem_120 = None
        mul_tensor_302: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_dims_168, 1.0025575447570332);  squeeze_dims_168 = None
        mul_tensor_303: "f32[80]" = torch.ops.aten.mul.Tensor(mul_tensor_302, 0.1);  mul_tensor_302 = None
        mul_tensor_304: "f32[80]" = torch.ops.aten.mul.Tensor(arg384_1, 0.9)
        add_tensor_182: "f32[80]" = torch.ops.aten.add.Tensor(mul_tensor_303, mul_tensor_304);  mul_tensor_303 = mul_tensor_304 = None
        add_tensor_183: "i64[]" = torch.ops.aten.add.Tensor(arg388_1, 1)
        squeeze_dims_169: "f32[112]" = torch.ops.aten.squeeze.dims(getitem_123, [0, 2, 3]);  getitem_123 = None
        squeeze_dims_170: "f32[112]" = torch.ops.aten.squeeze.dims(rsqrt_61, [0, 2, 3]);  rsqrt_61 = None
        mul_tensor_305: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_dims_169, 0.1)
        mul_tensor_306: "f32[112]" = torch.ops.aten.mul.Tensor(arg389_1, 0.9)
        add_tensor_184: "f32[112]" = torch.ops.aten.add.Tensor(mul_tensor_305, mul_tensor_306);  mul_tensor_305 = mul_tensor_306 = None
        squeeze_dims_171: "f32[112]" = torch.ops.aten.squeeze.dims(getitem_122, [0, 2, 3]);  getitem_122 = None
        mul_tensor_307: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_dims_171, 1.0025575447570332);  squeeze_dims_171 = None
        mul_tensor_308: "f32[112]" = torch.ops.aten.mul.Tensor(mul_tensor_307, 0.1);  mul_tensor_307 = None
        mul_tensor_309: "f32[112]" = torch.ops.aten.mul.Tensor(arg390_1, 0.9)
        add_tensor_185: "f32[112]" = torch.ops.aten.add.Tensor(mul_tensor_308, mul_tensor_309);  mul_tensor_308 = mul_tensor_309 = None
        add_tensor_186: "i64[]" = torch.ops.aten.add.Tensor(arg394_1, 1)
        squeeze_dims_172: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_125, [0, 2, 3]);  getitem_125 = None
        squeeze_dims_173: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_62, [0, 2, 3]);  rsqrt_62 = None
        mul_tensor_310: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_dims_172, 0.1)
        mul_tensor_311: "f32[160]" = torch.ops.aten.mul.Tensor(arg395_1, 0.9)
        add_tensor_187: "f32[160]" = torch.ops.aten.add.Tensor(mul_tensor_310, mul_tensor_311);  mul_tensor_310 = mul_tensor_311 = None
        squeeze_dims_174: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_124, [0, 2, 3]);  getitem_124 = None
        mul_tensor_312: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_dims_174, 1.0025575447570332);  squeeze_dims_174 = None
        mul_tensor_313: "f32[160]" = torch.ops.aten.mul.Tensor(mul_tensor_312, 0.1);  mul_tensor_312 = None
        mul_tensor_314: "f32[160]" = torch.ops.aten.mul.Tensor(arg396_1, 0.9)
        add_tensor_188: "f32[160]" = torch.ops.aten.add.Tensor(mul_tensor_313, mul_tensor_314);  mul_tensor_313 = mul_tensor_314 = None
        add_tensor_189: "i64[]" = torch.ops.aten.add.Tensor(arg400_1, 1)
        squeeze_dims_175: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_127, [0, 2, 3]);  getitem_127 = None
        squeeze_dims_176: "f32[480]" = torch.ops.aten.squeeze.dims(rsqrt_63, [0, 2, 3]);  rsqrt_63 = None
        mul_tensor_315: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_dims_175, 0.1)
        mul_tensor_316: "f32[480]" = torch.ops.aten.mul.Tensor(arg401_1, 0.9)
        add_tensor_190: "f32[480]" = torch.ops.aten.add.Tensor(mul_tensor_315, mul_tensor_316);  mul_tensor_315 = mul_tensor_316 = None
        squeeze_dims_177: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_126, [0, 2, 3]);  getitem_126 = None
        mul_tensor_317: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_dims_177, 1.0025575447570332);  squeeze_dims_177 = None
        mul_tensor_318: "f32[480]" = torch.ops.aten.mul.Tensor(mul_tensor_317, 0.1);  mul_tensor_317 = None
        mul_tensor_319: "f32[480]" = torch.ops.aten.mul.Tensor(arg402_1, 0.9)
        add_tensor_191: "f32[480]" = torch.ops.aten.add.Tensor(mul_tensor_318, mul_tensor_319);  mul_tensor_318 = mul_tensor_319 = None
        add_tensor_192: "i64[]" = torch.ops.aten.add.Tensor(arg406_1, 1)
        squeeze_dims_178: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_129, [0, 2, 3]);  getitem_129 = None
        mul_tensor_320: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_dims_178, 0.1);  squeeze_dims_178 = None
        mul_tensor_321: "f32[480]" = torch.ops.aten.mul.Tensor(arg407_1, 0.9)
        add_tensor_193: "f32[480]" = torch.ops.aten.add.Tensor(mul_tensor_320, mul_tensor_321);  mul_tensor_320 = mul_tensor_321 = None
        squeeze_dims_179: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_128, [0, 2, 3]);  getitem_128 = None
        mul_tensor_322: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_dims_179, 1.0025575447570332);  squeeze_dims_179 = None
        mul_tensor_323: "f32[480]" = torch.ops.aten.mul.Tensor(mul_tensor_322, 0.1);  mul_tensor_322 = None
        mul_tensor_324: "f32[480]" = torch.ops.aten.mul.Tensor(arg408_1, 0.9)
        add_tensor_194: "f32[480]" = torch.ops.aten.add.Tensor(mul_tensor_323, mul_tensor_324);  mul_tensor_323 = mul_tensor_324 = None
        add_tensor_195: "i64[]" = torch.ops.aten.add.Tensor(arg412_1, 1)
        squeeze_dims_180: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_131, [0, 2, 3]);  getitem_131 = None
        squeeze_dims_181: "f32[80]" = torch.ops.aten.squeeze.dims(rsqrt_65, [0, 2, 3]);  rsqrt_65 = None
        mul_tensor_325: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_dims_180, 0.1)
        mul_tensor_326: "f32[80]" = torch.ops.aten.mul.Tensor(arg413_1, 0.9)
        add_tensor_196: "f32[80]" = torch.ops.aten.add.Tensor(mul_tensor_325, mul_tensor_326);  mul_tensor_325 = mul_tensor_326 = None
        squeeze_dims_182: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_130, [0, 2, 3]);  getitem_130 = None
        mul_tensor_327: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_dims_182, 1.0025575447570332);  squeeze_dims_182 = None
        mul_tensor_328: "f32[80]" = torch.ops.aten.mul.Tensor(mul_tensor_327, 0.1);  mul_tensor_327 = None
        mul_tensor_329: "f32[80]" = torch.ops.aten.mul.Tensor(arg414_1, 0.9)
        add_tensor_197: "f32[80]" = torch.ops.aten.add.Tensor(mul_tensor_328, mul_tensor_329);  mul_tensor_328 = mul_tensor_329 = None
        add_tensor_198: "i64[]" = torch.ops.aten.add.Tensor(arg418_1, 1)
        squeeze_dims_183: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_133, [0, 2, 3]);  getitem_133 = None
        squeeze_dims_184: "f32[80]" = torch.ops.aten.squeeze.dims(rsqrt_66, [0, 2, 3]);  rsqrt_66 = None
        mul_tensor_330: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_dims_183, 0.1)
        mul_tensor_331: "f32[80]" = torch.ops.aten.mul.Tensor(arg419_1, 0.9)
        add_tensor_199: "f32[80]" = torch.ops.aten.add.Tensor(mul_tensor_330, mul_tensor_331);  mul_tensor_330 = mul_tensor_331 = None
        squeeze_dims_185: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_132, [0, 2, 3]);  getitem_132 = None
        mul_tensor_332: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_dims_185, 1.0025575447570332);  squeeze_dims_185 = None
        mul_tensor_333: "f32[80]" = torch.ops.aten.mul.Tensor(mul_tensor_332, 0.1);  mul_tensor_332 = None
        mul_tensor_334: "f32[80]" = torch.ops.aten.mul.Tensor(arg420_1, 0.9)
        add_tensor_200: "f32[80]" = torch.ops.aten.add.Tensor(mul_tensor_333, mul_tensor_334);  mul_tensor_333 = mul_tensor_334 = None
        add_tensor_201: "i64[]" = torch.ops.aten.add.Tensor(arg424_1, 1)
        squeeze_dims_186: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_135, [0, 2, 3]);  getitem_135 = None
        squeeze_dims_187: "f32[480]" = torch.ops.aten.squeeze.dims(rsqrt_67, [0, 2, 3]);  rsqrt_67 = None
        mul_tensor_335: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_dims_186, 0.1)
        mul_tensor_336: "f32[480]" = torch.ops.aten.mul.Tensor(arg425_1, 0.9)
        add_tensor_202: "f32[480]" = torch.ops.aten.add.Tensor(mul_tensor_335, mul_tensor_336);  mul_tensor_335 = mul_tensor_336 = None
        squeeze_dims_188: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_134, [0, 2, 3]);  getitem_134 = None
        mul_tensor_337: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_dims_188, 1.0025575447570332);  squeeze_dims_188 = None
        mul_tensor_338: "f32[480]" = torch.ops.aten.mul.Tensor(mul_tensor_337, 0.1);  mul_tensor_337 = None
        mul_tensor_339: "f32[480]" = torch.ops.aten.mul.Tensor(arg426_1, 0.9)
        add_tensor_203: "f32[480]" = torch.ops.aten.add.Tensor(mul_tensor_338, mul_tensor_339);  mul_tensor_338 = mul_tensor_339 = None
        add_tensor_204: "i64[]" = torch.ops.aten.add.Tensor(arg430_1, 1)
        squeeze_dims_189: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_137, [0, 2, 3]);  getitem_137 = None
        mul_tensor_340: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_dims_189, 0.1);  squeeze_dims_189 = None
        mul_tensor_341: "f32[480]" = torch.ops.aten.mul.Tensor(arg431_1, 0.9)
        add_tensor_205: "f32[480]" = torch.ops.aten.add.Tensor(mul_tensor_340, mul_tensor_341);  mul_tensor_340 = mul_tensor_341 = None
        squeeze_dims_190: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_136, [0, 2, 3]);  getitem_136 = None
        mul_tensor_342: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_dims_190, 1.0025575447570332);  squeeze_dims_190 = None
        mul_tensor_343: "f32[480]" = torch.ops.aten.mul.Tensor(mul_tensor_342, 0.1);  mul_tensor_342 = None
        mul_tensor_344: "f32[480]" = torch.ops.aten.mul.Tensor(arg432_1, 0.9)
        add_tensor_206: "f32[480]" = torch.ops.aten.add.Tensor(mul_tensor_343, mul_tensor_344);  mul_tensor_343 = mul_tensor_344 = None
        add_tensor_207: "i64[]" = torch.ops.aten.add.Tensor(arg440_1, 1)
        squeeze_dims_191: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_139, [0, 2, 3]);  getitem_139 = None
        squeeze_dims_192: "f32[80]" = torch.ops.aten.squeeze.dims(rsqrt_69, [0, 2, 3]);  rsqrt_69 = None
        mul_tensor_345: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_dims_191, 0.1)
        mul_tensor_346: "f32[80]" = torch.ops.aten.mul.Tensor(arg441_1, 0.9)
        add_tensor_208: "f32[80]" = torch.ops.aten.add.Tensor(mul_tensor_345, mul_tensor_346);  mul_tensor_345 = mul_tensor_346 = None
        squeeze_dims_193: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_138, [0, 2, 3]);  getitem_138 = None
        mul_tensor_347: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_dims_193, 1.0025575447570332);  squeeze_dims_193 = None
        mul_tensor_348: "f32[80]" = torch.ops.aten.mul.Tensor(mul_tensor_347, 0.1);  mul_tensor_347 = None
        mul_tensor_349: "f32[80]" = torch.ops.aten.mul.Tensor(arg442_1, 0.9)
        add_tensor_209: "f32[80]" = torch.ops.aten.add.Tensor(mul_tensor_348, mul_tensor_349);  mul_tensor_348 = mul_tensor_349 = None
        add_tensor_210: "i64[]" = torch.ops.aten.add.Tensor(arg446_1, 1)
        squeeze_dims_194: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_141, [0, 2, 3]);  getitem_141 = None
        squeeze_dims_195: "f32[80]" = torch.ops.aten.squeeze.dims(rsqrt_70, [0, 2, 3]);  rsqrt_70 = None
        mul_tensor_350: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_dims_194, 0.1)
        mul_tensor_351: "f32[80]" = torch.ops.aten.mul.Tensor(arg447_1, 0.9)
        add_tensor_211: "f32[80]" = torch.ops.aten.add.Tensor(mul_tensor_350, mul_tensor_351);  mul_tensor_350 = mul_tensor_351 = None
        squeeze_dims_196: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_140, [0, 2, 3]);  getitem_140 = None
        mul_tensor_352: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_dims_196, 1.0025575447570332);  squeeze_dims_196 = None
        mul_tensor_353: "f32[80]" = torch.ops.aten.mul.Tensor(mul_tensor_352, 0.1);  mul_tensor_352 = None
        mul_tensor_354: "f32[80]" = torch.ops.aten.mul.Tensor(arg448_1, 0.9)
        add_tensor_212: "f32[80]" = torch.ops.aten.add.Tensor(mul_tensor_353, mul_tensor_354);  mul_tensor_353 = mul_tensor_354 = None
        add_tensor_213: "i64[]" = torch.ops.aten.add.Tensor(arg452_1, 1)
        squeeze_dims_197: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_143, [0, 2, 3]);  getitem_143 = None
        squeeze_dims_198: "f32[480]" = torch.ops.aten.squeeze.dims(rsqrt_71, [0, 2, 3]);  rsqrt_71 = None
        mul_tensor_355: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_dims_197, 0.1)
        mul_tensor_356: "f32[480]" = torch.ops.aten.mul.Tensor(arg453_1, 0.9)
        add_tensor_214: "f32[480]" = torch.ops.aten.add.Tensor(mul_tensor_355, mul_tensor_356);  mul_tensor_355 = mul_tensor_356 = None
        squeeze_dims_199: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_142, [0, 2, 3]);  getitem_142 = None
        mul_tensor_357: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_dims_199, 1.0025575447570332);  squeeze_dims_199 = None
        mul_tensor_358: "f32[480]" = torch.ops.aten.mul.Tensor(mul_tensor_357, 0.1);  mul_tensor_357 = None
        mul_tensor_359: "f32[480]" = torch.ops.aten.mul.Tensor(arg454_1, 0.9)
        add_tensor_215: "f32[480]" = torch.ops.aten.add.Tensor(mul_tensor_358, mul_tensor_359);  mul_tensor_358 = mul_tensor_359 = None
        add_tensor_216: "i64[]" = torch.ops.aten.add.Tensor(arg458_1, 1)
        squeeze_dims_200: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_145, [0, 2, 3]);  getitem_145 = None
        mul_tensor_360: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_dims_200, 0.1);  squeeze_dims_200 = None
        mul_tensor_361: "f32[480]" = torch.ops.aten.mul.Tensor(arg459_1, 0.9)
        add_tensor_217: "f32[480]" = torch.ops.aten.add.Tensor(mul_tensor_360, mul_tensor_361);  mul_tensor_360 = mul_tensor_361 = None
        squeeze_dims_201: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_144, [0, 2, 3]);  getitem_144 = None
        mul_tensor_362: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_dims_201, 1.0025575447570332);  squeeze_dims_201 = None
        mul_tensor_363: "f32[480]" = torch.ops.aten.mul.Tensor(mul_tensor_362, 0.1);  mul_tensor_362 = None
        mul_tensor_364: "f32[480]" = torch.ops.aten.mul.Tensor(arg460_1, 0.9)
        add_tensor_218: "f32[480]" = torch.ops.aten.add.Tensor(mul_tensor_363, mul_tensor_364);  mul_tensor_363 = mul_tensor_364 = None
        add_tensor_219: "i64[]" = torch.ops.aten.add.Tensor(arg464_1, 1)
        squeeze_dims_202: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_147, [0, 2, 3]);  getitem_147 = None
        squeeze_dims_203: "f32[80]" = torch.ops.aten.squeeze.dims(rsqrt_73, [0, 2, 3]);  rsqrt_73 = None
        mul_tensor_365: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_dims_202, 0.1)
        mul_tensor_366: "f32[80]" = torch.ops.aten.mul.Tensor(arg465_1, 0.9)
        add_tensor_220: "f32[80]" = torch.ops.aten.add.Tensor(mul_tensor_365, mul_tensor_366);  mul_tensor_365 = mul_tensor_366 = None
        squeeze_dims_204: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_146, [0, 2, 3]);  getitem_146 = None
        mul_tensor_367: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_dims_204, 1.0025575447570332);  squeeze_dims_204 = None
        mul_tensor_368: "f32[80]" = torch.ops.aten.mul.Tensor(mul_tensor_367, 0.1);  mul_tensor_367 = None
        mul_tensor_369: "f32[80]" = torch.ops.aten.mul.Tensor(arg466_1, 0.9)
        add_tensor_221: "f32[80]" = torch.ops.aten.add.Tensor(mul_tensor_368, mul_tensor_369);  mul_tensor_368 = mul_tensor_369 = None
        add_tensor_222: "i64[]" = torch.ops.aten.add.Tensor(arg470_1, 1)
        squeeze_dims_205: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_149, [0, 2, 3]);  getitem_149 = None
        squeeze_dims_206: "f32[80]" = torch.ops.aten.squeeze.dims(rsqrt_74, [0, 2, 3]);  rsqrt_74 = None
        mul_tensor_370: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_dims_205, 0.1)
        mul_tensor_371: "f32[80]" = torch.ops.aten.mul.Tensor(arg471_1, 0.9)
        add_tensor_223: "f32[80]" = torch.ops.aten.add.Tensor(mul_tensor_370, mul_tensor_371);  mul_tensor_370 = mul_tensor_371 = None
        squeeze_dims_207: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_148, [0, 2, 3]);  getitem_148 = None
        mul_tensor_372: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_dims_207, 1.0025575447570332);  squeeze_dims_207 = None
        mul_tensor_373: "f32[80]" = torch.ops.aten.mul.Tensor(mul_tensor_372, 0.1);  mul_tensor_372 = None
        mul_tensor_374: "f32[80]" = torch.ops.aten.mul.Tensor(arg472_1, 0.9)
        add_tensor_224: "f32[80]" = torch.ops.aten.add.Tensor(mul_tensor_373, mul_tensor_374);  mul_tensor_373 = mul_tensor_374 = None
        add_tensor_225: "i64[]" = torch.ops.aten.add.Tensor(arg476_1, 1)
        squeeze_dims_208: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_151, [0, 2, 3]);  getitem_151 = None
        squeeze_dims_209: "f32[480]" = torch.ops.aten.squeeze.dims(rsqrt_75, [0, 2, 3]);  rsqrt_75 = None
        mul_tensor_375: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_dims_208, 0.1)
        mul_tensor_376: "f32[480]" = torch.ops.aten.mul.Tensor(arg477_1, 0.9)
        add_tensor_226: "f32[480]" = torch.ops.aten.add.Tensor(mul_tensor_375, mul_tensor_376);  mul_tensor_375 = mul_tensor_376 = None
        squeeze_dims_210: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_150, [0, 2, 3]);  getitem_150 = None
        mul_tensor_377: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_dims_210, 1.0025575447570332);  squeeze_dims_210 = None
        mul_tensor_378: "f32[480]" = torch.ops.aten.mul.Tensor(mul_tensor_377, 0.1);  mul_tensor_377 = None
        mul_tensor_379: "f32[480]" = torch.ops.aten.mul.Tensor(arg478_1, 0.9)
        add_tensor_227: "f32[480]" = torch.ops.aten.add.Tensor(mul_tensor_378, mul_tensor_379);  mul_tensor_378 = mul_tensor_379 = None
        add_tensor_228: "i64[]" = torch.ops.aten.add.Tensor(arg482_1, 1)
        squeeze_dims_211: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_153, [0, 2, 3]);  getitem_153 = None
        mul_tensor_380: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_dims_211, 0.1);  squeeze_dims_211 = None
        mul_tensor_381: "f32[480]" = torch.ops.aten.mul.Tensor(arg483_1, 0.9)
        add_tensor_229: "f32[480]" = torch.ops.aten.add.Tensor(mul_tensor_380, mul_tensor_381);  mul_tensor_380 = mul_tensor_381 = None
        squeeze_dims_212: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_152, [0, 2, 3]);  getitem_152 = None
        mul_tensor_382: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_dims_212, 1.0025575447570332);  squeeze_dims_212 = None
        mul_tensor_383: "f32[480]" = torch.ops.aten.mul.Tensor(mul_tensor_382, 0.1);  mul_tensor_382 = None
        mul_tensor_384: "f32[480]" = torch.ops.aten.mul.Tensor(arg484_1, 0.9)
        add_tensor_230: "f32[480]" = torch.ops.aten.add.Tensor(mul_tensor_383, mul_tensor_384);  mul_tensor_383 = mul_tensor_384 = None
        add_tensor_231: "i64[]" = torch.ops.aten.add.Tensor(arg492_1, 1)
        squeeze_dims_213: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_155, [0, 2, 3]);  getitem_155 = None
        squeeze_dims_214: "f32[80]" = torch.ops.aten.squeeze.dims(rsqrt_77, [0, 2, 3]);  rsqrt_77 = None
        mul_tensor_385: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_dims_213, 0.1)
        mul_tensor_386: "f32[80]" = torch.ops.aten.mul.Tensor(arg493_1, 0.9)
        add_tensor_232: "f32[80]" = torch.ops.aten.add.Tensor(mul_tensor_385, mul_tensor_386);  mul_tensor_385 = mul_tensor_386 = None
        squeeze_dims_215: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_154, [0, 2, 3]);  getitem_154 = None
        mul_tensor_387: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_dims_215, 1.0025575447570332);  squeeze_dims_215 = None
        mul_tensor_388: "f32[80]" = torch.ops.aten.mul.Tensor(mul_tensor_387, 0.1);  mul_tensor_387 = None
        mul_tensor_389: "f32[80]" = torch.ops.aten.mul.Tensor(arg494_1, 0.9)
        add_tensor_233: "f32[80]" = torch.ops.aten.add.Tensor(mul_tensor_388, mul_tensor_389);  mul_tensor_388 = mul_tensor_389 = None
        add_tensor_234: "i64[]" = torch.ops.aten.add.Tensor(arg498_1, 1)
        squeeze_dims_216: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_157, [0, 2, 3]);  getitem_157 = None
        squeeze_dims_217: "f32[80]" = torch.ops.aten.squeeze.dims(rsqrt_78, [0, 2, 3]);  rsqrt_78 = None
        mul_tensor_390: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_dims_216, 0.1)
        mul_tensor_391: "f32[80]" = torch.ops.aten.mul.Tensor(arg499_1, 0.9)
        add_tensor_235: "f32[80]" = torch.ops.aten.add.Tensor(mul_tensor_390, mul_tensor_391);  mul_tensor_390 = mul_tensor_391 = None
        squeeze_dims_218: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_156, [0, 2, 3]);  getitem_156 = None
        mul_tensor_392: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_dims_218, 1.0025575447570332);  squeeze_dims_218 = None
        mul_tensor_393: "f32[80]" = torch.ops.aten.mul.Tensor(mul_tensor_392, 0.1);  mul_tensor_392 = None
        mul_tensor_394: "f32[80]" = torch.ops.aten.mul.Tensor(arg500_1, 0.9)
        add_tensor_236: "f32[80]" = torch.ops.aten.add.Tensor(mul_tensor_393, mul_tensor_394);  mul_tensor_393 = mul_tensor_394 = None
        add_tensor_237: "i64[]" = torch.ops.aten.add.Tensor(arg504_1, 1)
        squeeze_dims_219: "f32[960]" = torch.ops.aten.squeeze.dims(getitem_159, [0, 2, 3]);  getitem_159 = None
        mul_tensor_395: "f32[960]" = torch.ops.aten.mul.Tensor(squeeze_dims_219, 0.1);  squeeze_dims_219 = None
        mul_tensor_396: "f32[960]" = torch.ops.aten.mul.Tensor(arg505_1, 0.9)
        add_tensor_238: "f32[960]" = torch.ops.aten.add.Tensor(mul_tensor_395, mul_tensor_396);  mul_tensor_395 = mul_tensor_396 = None
        squeeze_dims_220: "f32[960]" = torch.ops.aten.squeeze.dims(getitem_158, [0, 2, 3]);  getitem_158 = None
        mul_tensor_397: "f32[960]" = torch.ops.aten.mul.Tensor(squeeze_dims_220, 1.0025575447570332);  squeeze_dims_220 = None
        mul_tensor_398: "f32[960]" = torch.ops.aten.mul.Tensor(mul_tensor_397, 0.1);  mul_tensor_397 = None
        mul_tensor_399: "f32[960]" = torch.ops.aten.mul.Tensor(arg506_1, 0.9)
        add_tensor_239: "f32[960]" = torch.ops.aten.add.Tensor(mul_tensor_398, mul_tensor_399);  mul_tensor_398 = mul_tensor_399 = None
        relu_default: "f16[8, 1280, 1, 1]" = torch.ops.aten.relu.default(convolution_94);  convolution_94 = None
        convert_element_type_default: "f16[1000]" = torch.ops.prims.convert_element_type.default(arg512_1, torch.float16);  arg512_1 = None
        convert_element_type_default_1: "f16[1000, 1280]" = torch.ops.prims.convert_element_type.default(arg511_1, torch.float16);  arg511_1 = None
        permute_default: "f16[1280, 1000]" = torch.ops.aten.permute.default(convert_element_type_default_1, [1, 0]);  convert_element_type_default_1 = None
        view_default: "f16[8, 1280]" = torch.ops.aten.view.default(relu_default, _shape_param_0);  _shape_param_0 = None
        permute_default_1: "f16[1000, 1280]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        le_scalar: "b8[8, 1280, 1, 1]" = torch.ops.aten.le.Scalar(relu_default, 0);  relu_default = None
        unsqueeze_default: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(squeeze_dims_216, 0);  squeeze_dims_216 = None
        unsqueeze_default_1: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        unsqueeze_default_3: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(squeeze_dims_213, 0);  squeeze_dims_213 = None
        unsqueeze_default_4: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        unsqueeze_default_6: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(squeeze_dims_208, 0);  squeeze_dims_208 = None
        unsqueeze_default_7: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        unsqueeze_default_9: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(squeeze_dims_205, 0);  squeeze_dims_205 = None
        unsqueeze_default_10: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_9, 2);  unsqueeze_default_9 = None
        unsqueeze_default_11: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 3);  unsqueeze_default_10 = None
        unsqueeze_default_12: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(squeeze_dims_202, 0);  squeeze_dims_202 = None
        unsqueeze_default_13: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_12, 2);  unsqueeze_default_12 = None
        unsqueeze_default_14: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 3);  unsqueeze_default_13 = None
        unsqueeze_default_15: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(squeeze_dims_197, 0);  squeeze_dims_197 = None
        unsqueeze_default_16: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_15, 2);  unsqueeze_default_15 = None
        unsqueeze_default_17: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_16, 3);  unsqueeze_default_16 = None
        unsqueeze_default_18: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(squeeze_dims_194, 0);  squeeze_dims_194 = None
        unsqueeze_default_19: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_18, 2);  unsqueeze_default_18 = None
        unsqueeze_default_20: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_19, 3);  unsqueeze_default_19 = None
        unsqueeze_default_21: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(squeeze_dims_191, 0);  squeeze_dims_191 = None
        unsqueeze_default_22: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_21, 2);  unsqueeze_default_21 = None
        unsqueeze_default_23: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_22, 3);  unsqueeze_default_22 = None
        unsqueeze_default_24: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(squeeze_dims_186, 0);  squeeze_dims_186 = None
        unsqueeze_default_25: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_24, 2);  unsqueeze_default_24 = None
        unsqueeze_default_26: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_25, 3);  unsqueeze_default_25 = None
        unsqueeze_default_27: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(squeeze_dims_183, 0);  squeeze_dims_183 = None
        unsqueeze_default_28: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_27, 2);  unsqueeze_default_27 = None
        unsqueeze_default_29: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_28, 3);  unsqueeze_default_28 = None
        unsqueeze_default_30: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(squeeze_dims_180, 0);  squeeze_dims_180 = None
        unsqueeze_default_31: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_30, 2);  unsqueeze_default_30 = None
        unsqueeze_default_32: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_31, 3);  unsqueeze_default_31 = None
        unsqueeze_default_33: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(squeeze_dims_175, 0);  squeeze_dims_175 = None
        unsqueeze_default_34: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_33, 2);  unsqueeze_default_33 = None
        unsqueeze_default_35: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_34, 3);  unsqueeze_default_34 = None
        unsqueeze_default_36: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_dims_172, 0);  squeeze_dims_172 = None
        unsqueeze_default_37: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_36, 2);  unsqueeze_default_36 = None
        unsqueeze_default_38: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_37, 3);  unsqueeze_default_37 = None
        unsqueeze_default_39: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(squeeze_dims_169, 0);  squeeze_dims_169 = None
        unsqueeze_default_40: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_39, 2);  unsqueeze_default_39 = None
        unsqueeze_default_41: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_40, 3);  unsqueeze_default_40 = None
        unsqueeze_default_42: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(squeeze_dims_166, 0);  squeeze_dims_166 = None
        unsqueeze_default_43: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_42, 2);  unsqueeze_default_42 = None
        unsqueeze_default_44: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_43, 3);  unsqueeze_default_43 = None
        unsqueeze_default_45: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(squeeze_dims_163, 0);  squeeze_dims_163 = None
        unsqueeze_default_46: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_45, 2);  unsqueeze_default_45 = None
        unsqueeze_default_47: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_46, 3);  unsqueeze_default_46 = None
        unsqueeze_default_48: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(squeeze_dims_156, 0);  squeeze_dims_156 = None
        unsqueeze_default_49: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_48, 2);  unsqueeze_default_48 = None
        unsqueeze_default_50: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_49, 3);  unsqueeze_default_49 = None
        unsqueeze_default_51: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(squeeze_dims_153, 0);  squeeze_dims_153 = None
        unsqueeze_default_52: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_51, 2);  unsqueeze_default_51 = None
        unsqueeze_default_53: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_52, 3);  unsqueeze_default_52 = None
        unsqueeze_default_54: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(squeeze_dims_150, 0);  squeeze_dims_150 = None
        unsqueeze_default_55: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_54, 2);  unsqueeze_default_54 = None
        unsqueeze_default_56: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_55, 3);  unsqueeze_default_55 = None
        unsqueeze_default_57: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(squeeze_dims_145, 0);  squeeze_dims_145 = None
        unsqueeze_default_58: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_57, 2);  unsqueeze_default_57 = None
        unsqueeze_default_59: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_58, 3);  unsqueeze_default_58 = None
        unsqueeze_default_60: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(squeeze_dims_142, 0);  squeeze_dims_142 = None
        unsqueeze_default_61: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_60, 2);  unsqueeze_default_60 = None
        unsqueeze_default_62: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_61, 3);  unsqueeze_default_61 = None
        unsqueeze_default_63: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(squeeze_dims_139, 0);  squeeze_dims_139 = None
        unsqueeze_default_64: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_63, 2);  unsqueeze_default_63 = None
        unsqueeze_default_65: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_64, 3);  unsqueeze_default_64 = None
        unsqueeze_default_66: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(squeeze_dims_136, 0);  squeeze_dims_136 = None
        unsqueeze_default_67: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_66, 2);  unsqueeze_default_66 = None
        unsqueeze_default_68: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_67, 3);  unsqueeze_default_67 = None
        unsqueeze_default_69: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(squeeze_dims_133, 0);  squeeze_dims_133 = None
        unsqueeze_default_70: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_69, 2);  unsqueeze_default_69 = None
        unsqueeze_default_71: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_70, 3);  unsqueeze_default_70 = None
        unsqueeze_default_72: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(squeeze_dims_128, 0);  squeeze_dims_128 = None
        unsqueeze_default_73: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_72, 2);  unsqueeze_default_72 = None
        unsqueeze_default_74: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_73, 3);  unsqueeze_default_73 = None
        unsqueeze_default_75: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(squeeze_dims_125, 0);  squeeze_dims_125 = None
        unsqueeze_default_76: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_75, 2);  unsqueeze_default_75 = None
        unsqueeze_default_77: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_76, 3);  unsqueeze_default_76 = None
        unsqueeze_default_78: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(squeeze_dims_122, 0);  squeeze_dims_122 = None
        unsqueeze_default_79: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_78, 2);  unsqueeze_default_78 = None
        unsqueeze_default_80: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_79, 3);  unsqueeze_default_79 = None
        unsqueeze_default_81: "f32[1, 92]" = torch.ops.aten.unsqueeze.default(squeeze_dims_117, 0);  squeeze_dims_117 = None
        unsqueeze_default_82: "f32[1, 92, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_81, 2);  unsqueeze_default_81 = None
        unsqueeze_default_83: "f32[1, 92, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_82, 3);  unsqueeze_default_82 = None
        unsqueeze_default_84: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(squeeze_dims_114, 0);  squeeze_dims_114 = None
        unsqueeze_default_85: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_84, 2);  unsqueeze_default_84 = None
        unsqueeze_default_86: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_85, 3);  unsqueeze_default_85 = None
        unsqueeze_default_87: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(squeeze_dims_111, 0);  squeeze_dims_111 = None
        unsqueeze_default_88: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_87, 2);  unsqueeze_default_87 = None
        unsqueeze_default_89: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_88, 3);  unsqueeze_default_88 = None
        unsqueeze_default_90: "f32[1, 92]" = torch.ops.aten.unsqueeze.default(squeeze_dims_106, 0);  squeeze_dims_106 = None
        unsqueeze_default_91: "f32[1, 92, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_90, 2);  unsqueeze_default_90 = None
        unsqueeze_default_92: "f32[1, 92, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_91, 3);  unsqueeze_default_91 = None
        unsqueeze_default_93: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(squeeze_dims_103, 0);  squeeze_dims_103 = None
        unsqueeze_default_94: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_93, 2);  unsqueeze_default_93 = None
        unsqueeze_default_95: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_94, 3);  unsqueeze_default_94 = None
        unsqueeze_default_96: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(squeeze_dims_100, 0);  squeeze_dims_100 = None
        unsqueeze_default_97: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_96, 2);  unsqueeze_default_96 = None
        unsqueeze_default_98: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_97, 3);  unsqueeze_default_97 = None
        unsqueeze_default_99: "f32[1, 100]" = torch.ops.aten.unsqueeze.default(squeeze_dims_95, 0);  squeeze_dims_95 = None
        unsqueeze_default_100: "f32[1, 100, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_99, 2);  unsqueeze_default_99 = None
        unsqueeze_default_101: "f32[1, 100, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_100, 3);  unsqueeze_default_100 = None
        unsqueeze_default_102: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(squeeze_dims_92, 0);  squeeze_dims_92 = None
        unsqueeze_default_103: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_102, 2);  unsqueeze_default_102 = None
        unsqueeze_default_104: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_103, 3);  unsqueeze_default_103 = None
        unsqueeze_default_105: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(squeeze_dims_89, 0);  squeeze_dims_89 = None
        unsqueeze_default_106: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_105, 2);  unsqueeze_default_105 = None
        unsqueeze_default_107: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_106, 3);  unsqueeze_default_106 = None
        unsqueeze_default_108: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(squeeze_dims_86, 0);  squeeze_dims_86 = None
        unsqueeze_default_109: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_108, 2);  unsqueeze_default_108 = None
        unsqueeze_default_110: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_109, 3);  unsqueeze_default_109 = None
        unsqueeze_default_111: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(squeeze_dims_83, 0);  squeeze_dims_83 = None
        unsqueeze_default_112: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_111, 2);  unsqueeze_default_111 = None
        unsqueeze_default_113: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_112, 3);  unsqueeze_default_112 = None
        unsqueeze_default_114: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(squeeze_dims_80, 0);  squeeze_dims_80 = None
        unsqueeze_default_115: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_114, 2);  unsqueeze_default_114 = None
        unsqueeze_default_116: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_115, 3);  unsqueeze_default_115 = None
        unsqueeze_default_117: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(squeeze_dims_75, 0);  squeeze_dims_75 = None
        unsqueeze_default_118: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_117, 2);  unsqueeze_default_117 = None
        unsqueeze_default_119: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_118, 3);  unsqueeze_default_118 = None
        unsqueeze_default_120: "f32[1, 20]" = torch.ops.aten.unsqueeze.default(squeeze_dims_72, 0);  squeeze_dims_72 = None
        unsqueeze_default_121: "f32[1, 20, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_120, 2);  unsqueeze_default_120 = None
        unsqueeze_default_122: "f32[1, 20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_121, 3);  unsqueeze_default_121 = None
        unsqueeze_default_123: "f32[1, 20]" = torch.ops.aten.unsqueeze.default(squeeze_dims_69, 0);  squeeze_dims_69 = None
        unsqueeze_default_124: "f32[1, 20, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_123, 2);  unsqueeze_default_123 = None
        unsqueeze_default_125: "f32[1, 20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_124, 3);  unsqueeze_default_124 = None
        unsqueeze_default_126: "f32[1, 60]" = torch.ops.aten.unsqueeze.default(squeeze_dims_64, 0);  squeeze_dims_64 = None
        unsqueeze_default_127: "f32[1, 60, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_126, 2);  unsqueeze_default_126 = None
        unsqueeze_default_128: "f32[1, 60, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_127, 3);  unsqueeze_default_127 = None
        unsqueeze_default_129: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(squeeze_dims_61, 0);  squeeze_dims_61 = None
        unsqueeze_default_130: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_129, 2);  unsqueeze_default_129 = None
        unsqueeze_default_131: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_130, 3);  unsqueeze_default_130 = None
        unsqueeze_default_132: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(squeeze_dims_58, 0);  squeeze_dims_58 = None
        unsqueeze_default_133: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_132, 2);  unsqueeze_default_132 = None
        unsqueeze_default_134: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_133, 3);  unsqueeze_default_133 = None
        unsqueeze_default_135: "f32[1, 20]" = torch.ops.aten.unsqueeze.default(squeeze_dims_55, 0);  squeeze_dims_55 = None
        unsqueeze_default_136: "f32[1, 20, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_135, 2);  unsqueeze_default_135 = None
        unsqueeze_default_137: "f32[1, 20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_136, 3);  unsqueeze_default_136 = None
        unsqueeze_default_138: "f32[1, 20]" = torch.ops.aten.unsqueeze.default(squeeze_dims_52, 0);  squeeze_dims_52 = None
        unsqueeze_default_139: "f32[1, 20, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_138, 2);  unsqueeze_default_138 = None
        unsqueeze_default_140: "f32[1, 20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_139, 3);  unsqueeze_default_139 = None
        unsqueeze_default_141: "f32[1, 36]" = torch.ops.aten.unsqueeze.default(squeeze_dims_45, 0);  squeeze_dims_45 = None
        unsqueeze_default_142: "f32[1, 36, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_141, 2);  unsqueeze_default_141 = None
        unsqueeze_default_143: "f32[1, 36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_142, 3);  unsqueeze_default_142 = None
        unsqueeze_default_144: "f32[1, 12]" = torch.ops.aten.unsqueeze.default(squeeze_dims_42, 0);  squeeze_dims_42 = None
        unsqueeze_default_145: "f32[1, 12, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_144, 2);  unsqueeze_default_144 = None
        unsqueeze_default_146: "f32[1, 12, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_145, 3);  unsqueeze_default_145 = None
        unsqueeze_default_147: "f32[1, 12]" = torch.ops.aten.unsqueeze.default(squeeze_dims_39, 0);  squeeze_dims_39 = None
        unsqueeze_default_148: "f32[1, 12, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_147, 2);  unsqueeze_default_147 = None
        unsqueeze_default_149: "f32[1, 12, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_148, 3);  unsqueeze_default_148 = None
        unsqueeze_default_150: "f32[1, 36]" = torch.ops.aten.unsqueeze.default(squeeze_dims_34, 0);  squeeze_dims_34 = None
        unsqueeze_default_151: "f32[1, 36, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_150, 2);  unsqueeze_default_150 = None
        unsqueeze_default_152: "f32[1, 36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_151, 3);  unsqueeze_default_151 = None
        unsqueeze_default_153: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(squeeze_dims_31, 0);  squeeze_dims_31 = None
        unsqueeze_default_154: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_153, 2);  unsqueeze_default_153 = None
        unsqueeze_default_155: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_154, 3);  unsqueeze_default_154 = None
        unsqueeze_default_156: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(squeeze_dims_28, 0);  squeeze_dims_28 = None
        unsqueeze_default_157: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_156, 2);  unsqueeze_default_156 = None
        unsqueeze_default_158: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_157, 3);  unsqueeze_default_157 = None
        unsqueeze_default_159: "f32[1, 12]" = torch.ops.aten.unsqueeze.default(squeeze_dims_25, 0);  squeeze_dims_25 = None
        unsqueeze_default_160: "f32[1, 12, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_159, 2);  unsqueeze_default_159 = None
        unsqueeze_default_161: "f32[1, 12, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_160, 3);  unsqueeze_default_160 = None
        unsqueeze_default_162: "f32[1, 12]" = torch.ops.aten.unsqueeze.default(squeeze_dims_22, 0);  squeeze_dims_22 = None
        unsqueeze_default_163: "f32[1, 12, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_162, 2);  unsqueeze_default_162 = None
        unsqueeze_default_164: "f32[1, 12, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_163, 3);  unsqueeze_default_163 = None
        unsqueeze_default_165: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(squeeze_dims_19, 0);  squeeze_dims_19 = None
        unsqueeze_default_166: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_165, 2);  unsqueeze_default_165 = None
        unsqueeze_default_167: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_166, 3);  unsqueeze_default_166 = None
        unsqueeze_default_168: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(squeeze_dims_14, 0);  squeeze_dims_14 = None
        unsqueeze_default_169: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_168, 2);  unsqueeze_default_168 = None
        unsqueeze_default_170: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_169, 3);  unsqueeze_default_169 = None
        unsqueeze_default_171: "f32[1, 8]" = torch.ops.aten.unsqueeze.default(squeeze_dims_11, 0);  squeeze_dims_11 = None
        unsqueeze_default_172: "f32[1, 8, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_171, 2);  unsqueeze_default_171 = None
        unsqueeze_default_173: "f32[1, 8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_172, 3);  unsqueeze_default_172 = None
        unsqueeze_default_174: "f32[1, 8]" = torch.ops.aten.unsqueeze.default(squeeze_dims_8, 0);  squeeze_dims_8 = None
        unsqueeze_default_175: "f32[1, 8, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_174, 2);  unsqueeze_default_174 = None
        unsqueeze_default_176: "f32[1, 8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_175, 3);  unsqueeze_default_175 = None
        unsqueeze_default_177: "f32[1, 8]" = torch.ops.aten.unsqueeze.default(squeeze_dims_3, 0);  squeeze_dims_3 = None
        unsqueeze_default_178: "f32[1, 8, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_177, 2);  unsqueeze_default_177 = None
        unsqueeze_default_179: "f32[1, 8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_178, 3);  unsqueeze_default_178 = None
        unsqueeze_default_180: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_181: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_180, 2);  unsqueeze_default_180 = None
        unsqueeze_default_182: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_181, 3);  unsqueeze_default_181 = None
        copy__default: "i64[]" = torch.ops.aten.copy_.default(arg2_1, add_tensor);  arg2_1 = add_tensor = None
        copy__default_1: "f32[16]" = torch.ops.aten.copy_.default(arg3_1, add_tensor_1);  arg3_1 = add_tensor_1 = None
        copy__default_2: "f32[16]" = torch.ops.aten.copy_.default(arg4_1, add_tensor_2);  arg4_1 = add_tensor_2 = None
        copy__default_3: "i64[]" = torch.ops.aten.copy_.default(arg8_1, add_tensor_3);  arg8_1 = add_tensor_3 = None
        copy__default_4: "f32[8]" = torch.ops.aten.copy_.default(arg9_1, add_tensor_4);  arg9_1 = add_tensor_4 = None
        copy__default_5: "f32[8]" = torch.ops.aten.copy_.default(arg10_1, add_tensor_5);  arg10_1 = add_tensor_5 = None
        copy__default_6: "i64[]" = torch.ops.aten.copy_.default(arg14_1, add_tensor_6);  arg14_1 = add_tensor_6 = None
        copy__default_7: "f32[8]" = torch.ops.aten.copy_.default(arg15_1, add_tensor_7);  arg15_1 = add_tensor_7 = None
        copy__default_8: "f32[8]" = torch.ops.aten.copy_.default(arg16_1, add_tensor_8);  arg16_1 = add_tensor_8 = None
        copy__default_9: "i64[]" = torch.ops.aten.copy_.default(arg20_1, add_tensor_9);  arg20_1 = add_tensor_9 = None
        copy__default_10: "f32[8]" = torch.ops.aten.copy_.default(arg21_1, add_tensor_10);  arg21_1 = add_tensor_10 = None
        copy__default_11: "f32[8]" = torch.ops.aten.copy_.default(arg22_1, add_tensor_11);  arg22_1 = add_tensor_11 = None
        copy__default_12: "i64[]" = torch.ops.aten.copy_.default(arg26_1, add_tensor_12);  arg26_1 = add_tensor_12 = None
        copy__default_13: "f32[8]" = torch.ops.aten.copy_.default(arg27_1, add_tensor_13);  arg27_1 = add_tensor_13 = None
        copy__default_14: "f32[8]" = torch.ops.aten.copy_.default(arg28_1, add_tensor_14);  arg28_1 = add_tensor_14 = None
        copy__default_15: "i64[]" = torch.ops.aten.copy_.default(arg32_1, add_tensor_15);  arg32_1 = add_tensor_15 = None
        copy__default_16: "f32[24]" = torch.ops.aten.copy_.default(arg33_1, add_tensor_16);  arg33_1 = add_tensor_16 = None
        copy__default_17: "f32[24]" = torch.ops.aten.copy_.default(arg34_1, add_tensor_17);  arg34_1 = add_tensor_17 = None
        copy__default_18: "i64[]" = torch.ops.aten.copy_.default(arg38_1, add_tensor_18);  arg38_1 = add_tensor_18 = None
        copy__default_19: "f32[24]" = torch.ops.aten.copy_.default(arg39_1, add_tensor_19);  arg39_1 = add_tensor_19 = None
        copy__default_20: "f32[24]" = torch.ops.aten.copy_.default(arg40_1, add_tensor_20);  arg40_1 = add_tensor_20 = None
        copy__default_21: "i64[]" = torch.ops.aten.copy_.default(arg44_1, add_tensor_21);  arg44_1 = add_tensor_21 = None
        copy__default_22: "f32[48]" = torch.ops.aten.copy_.default(arg45_1, add_tensor_22);  arg45_1 = add_tensor_22 = None
        copy__default_23: "f32[48]" = torch.ops.aten.copy_.default(arg46_1, add_tensor_23);  arg46_1 = add_tensor_23 = None
        copy__default_24: "i64[]" = torch.ops.aten.copy_.default(arg50_1, add_tensor_24);  arg50_1 = add_tensor_24 = None
        copy__default_25: "f32[12]" = torch.ops.aten.copy_.default(arg51_1, add_tensor_25);  arg51_1 = add_tensor_25 = None
        copy__default_26: "f32[12]" = torch.ops.aten.copy_.default(arg52_1, add_tensor_26);  arg52_1 = add_tensor_26 = None
        copy__default_27: "i64[]" = torch.ops.aten.copy_.default(arg56_1, add_tensor_27);  arg56_1 = add_tensor_27 = None
        copy__default_28: "f32[12]" = torch.ops.aten.copy_.default(arg57_1, add_tensor_28);  arg57_1 = add_tensor_28 = None
        copy__default_29: "f32[12]" = torch.ops.aten.copy_.default(arg58_1, add_tensor_29);  arg58_1 = add_tensor_29 = None
        copy__default_30: "i64[]" = torch.ops.aten.copy_.default(arg62_1, add_tensor_30);  arg62_1 = add_tensor_30 = None
        copy__default_31: "f32[16]" = torch.ops.aten.copy_.default(arg63_1, add_tensor_31);  arg63_1 = add_tensor_31 = None
        copy__default_32: "f32[16]" = torch.ops.aten.copy_.default(arg64_1, add_tensor_32);  arg64_1 = add_tensor_32 = None
        copy__default_33: "i64[]" = torch.ops.aten.copy_.default(arg68_1, add_tensor_33);  arg68_1 = add_tensor_33 = None
        copy__default_34: "f32[24]" = torch.ops.aten.copy_.default(arg69_1, add_tensor_34);  arg69_1 = add_tensor_34 = None
        copy__default_35: "f32[24]" = torch.ops.aten.copy_.default(arg70_1, add_tensor_35);  arg70_1 = add_tensor_35 = None
        copy__default_36: "i64[]" = torch.ops.aten.copy_.default(arg74_1, add_tensor_36);  arg74_1 = add_tensor_36 = None
        copy__default_37: "f32[36]" = torch.ops.aten.copy_.default(arg75_1, add_tensor_37);  arg75_1 = add_tensor_37 = None
        copy__default_38: "f32[36]" = torch.ops.aten.copy_.default(arg76_1, add_tensor_38);  arg76_1 = add_tensor_38 = None
        copy__default_39: "i64[]" = torch.ops.aten.copy_.default(arg80_1, add_tensor_39);  arg80_1 = add_tensor_39 = None
        copy__default_40: "f32[36]" = torch.ops.aten.copy_.default(arg81_1, add_tensor_40);  arg81_1 = add_tensor_40 = None
        copy__default_41: "f32[36]" = torch.ops.aten.copy_.default(arg82_1, add_tensor_41);  arg82_1 = add_tensor_41 = None
        copy__default_42: "i64[]" = torch.ops.aten.copy_.default(arg86_1, add_tensor_42);  arg86_1 = add_tensor_42 = None
        copy__default_43: "f32[12]" = torch.ops.aten.copy_.default(arg87_1, add_tensor_43);  arg87_1 = add_tensor_43 = None
        copy__default_44: "f32[12]" = torch.ops.aten.copy_.default(arg88_1, add_tensor_44);  arg88_1 = add_tensor_44 = None
        copy__default_45: "i64[]" = torch.ops.aten.copy_.default(arg92_1, add_tensor_45);  arg92_1 = add_tensor_45 = None
        copy__default_46: "f32[12]" = torch.ops.aten.copy_.default(arg93_1, add_tensor_46);  arg93_1 = add_tensor_46 = None
        copy__default_47: "f32[12]" = torch.ops.aten.copy_.default(arg94_1, add_tensor_47);  arg94_1 = add_tensor_47 = None
        copy__default_48: "i64[]" = torch.ops.aten.copy_.default(arg98_1, add_tensor_48);  arg98_1 = add_tensor_48 = None
        copy__default_49: "f32[36]" = torch.ops.aten.copy_.default(arg99_1, add_tensor_49);  arg99_1 = add_tensor_49 = None
        copy__default_50: "f32[36]" = torch.ops.aten.copy_.default(arg100_1, add_tensor_50);  arg100_1 = add_tensor_50 = None
        copy__default_51: "i64[]" = torch.ops.aten.copy_.default(arg104_1, add_tensor_51);  arg104_1 = add_tensor_51 = None
        copy__default_52: "f32[36]" = torch.ops.aten.copy_.default(arg105_1, add_tensor_52);  arg105_1 = add_tensor_52 = None
        copy__default_53: "f32[36]" = torch.ops.aten.copy_.default(arg106_1, add_tensor_53);  arg106_1 = add_tensor_53 = None
        copy__default_54: "i64[]" = torch.ops.aten.copy_.default(arg110_1, add_tensor_54);  arg110_1 = add_tensor_54 = None
        copy__default_55: "f32[72]" = torch.ops.aten.copy_.default(arg111_1, add_tensor_55);  arg111_1 = add_tensor_55 = None
        copy__default_56: "f32[72]" = torch.ops.aten.copy_.default(arg112_1, add_tensor_56);  arg112_1 = add_tensor_56 = None
        copy__default_57: "i64[]" = torch.ops.aten.copy_.default(arg120_1, add_tensor_57);  arg120_1 = add_tensor_57 = None
        copy__default_58: "f32[20]" = torch.ops.aten.copy_.default(arg121_1, add_tensor_58);  arg121_1 = add_tensor_58 = None
        copy__default_59: "f32[20]" = torch.ops.aten.copy_.default(arg122_1, add_tensor_59);  arg122_1 = add_tensor_59 = None
        copy__default_60: "i64[]" = torch.ops.aten.copy_.default(arg126_1, add_tensor_60);  arg126_1 = add_tensor_60 = None
        copy__default_61: "f32[20]" = torch.ops.aten.copy_.default(arg127_1, add_tensor_61);  arg127_1 = add_tensor_61 = None
        copy__default_62: "f32[20]" = torch.ops.aten.copy_.default(arg128_1, add_tensor_62);  arg128_1 = add_tensor_62 = None
        copy__default_63: "i64[]" = torch.ops.aten.copy_.default(arg132_1, add_tensor_63);  arg132_1 = add_tensor_63 = None
        copy__default_64: "f32[24]" = torch.ops.aten.copy_.default(arg133_1, add_tensor_64);  arg133_1 = add_tensor_64 = None
        copy__default_65: "f32[24]" = torch.ops.aten.copy_.default(arg134_1, add_tensor_65);  arg134_1 = add_tensor_65 = None
        copy__default_66: "i64[]" = torch.ops.aten.copy_.default(arg138_1, add_tensor_66);  arg138_1 = add_tensor_66 = None
        copy__default_67: "f32[40]" = torch.ops.aten.copy_.default(arg139_1, add_tensor_67);  arg139_1 = add_tensor_67 = None
        copy__default_68: "f32[40]" = torch.ops.aten.copy_.default(arg140_1, add_tensor_68);  arg140_1 = add_tensor_68 = None
        copy__default_69: "i64[]" = torch.ops.aten.copy_.default(arg144_1, add_tensor_69);  arg144_1 = add_tensor_69 = None
        copy__default_70: "f32[60]" = torch.ops.aten.copy_.default(arg145_1, add_tensor_70);  arg145_1 = add_tensor_70 = None
        copy__default_71: "f32[60]" = torch.ops.aten.copy_.default(arg146_1, add_tensor_71);  arg146_1 = add_tensor_71 = None
        copy__default_72: "i64[]" = torch.ops.aten.copy_.default(arg150_1, add_tensor_72);  arg150_1 = add_tensor_72 = None
        copy__default_73: "f32[60]" = torch.ops.aten.copy_.default(arg151_1, add_tensor_73);  arg151_1 = add_tensor_73 = None
        copy__default_74: "f32[60]" = torch.ops.aten.copy_.default(arg152_1, add_tensor_74);  arg152_1 = add_tensor_74 = None
        copy__default_75: "i64[]" = torch.ops.aten.copy_.default(arg160_1, add_tensor_75);  arg160_1 = add_tensor_75 = None
        copy__default_76: "f32[20]" = torch.ops.aten.copy_.default(arg161_1, add_tensor_76);  arg161_1 = add_tensor_76 = None
        copy__default_77: "f32[20]" = torch.ops.aten.copy_.default(arg162_1, add_tensor_77);  arg162_1 = add_tensor_77 = None
        copy__default_78: "i64[]" = torch.ops.aten.copy_.default(arg166_1, add_tensor_78);  arg166_1 = add_tensor_78 = None
        copy__default_79: "f32[20]" = torch.ops.aten.copy_.default(arg167_1, add_tensor_79);  arg167_1 = add_tensor_79 = None
        copy__default_80: "f32[20]" = torch.ops.aten.copy_.default(arg168_1, add_tensor_80);  arg168_1 = add_tensor_80 = None
        copy__default_81: "i64[]" = torch.ops.aten.copy_.default(arg172_1, add_tensor_81);  arg172_1 = add_tensor_81 = None
        copy__default_82: "f32[120]" = torch.ops.aten.copy_.default(arg173_1, add_tensor_82);  arg173_1 = add_tensor_82 = None
        copy__default_83: "f32[120]" = torch.ops.aten.copy_.default(arg174_1, add_tensor_83);  arg174_1 = add_tensor_83 = None
        copy__default_84: "i64[]" = torch.ops.aten.copy_.default(arg178_1, add_tensor_84);  arg178_1 = add_tensor_84 = None
        copy__default_85: "f32[120]" = torch.ops.aten.copy_.default(arg179_1, add_tensor_85);  arg179_1 = add_tensor_85 = None
        copy__default_86: "f32[120]" = torch.ops.aten.copy_.default(arg180_1, add_tensor_86);  arg180_1 = add_tensor_86 = None
        copy__default_87: "i64[]" = torch.ops.aten.copy_.default(arg184_1, add_tensor_87);  arg184_1 = add_tensor_87 = None
        copy__default_88: "f32[240]" = torch.ops.aten.copy_.default(arg185_1, add_tensor_88);  arg185_1 = add_tensor_88 = None
        copy__default_89: "f32[240]" = torch.ops.aten.copy_.default(arg186_1, add_tensor_89);  arg186_1 = add_tensor_89 = None
        copy__default_90: "i64[]" = torch.ops.aten.copy_.default(arg190_1, add_tensor_90);  arg190_1 = add_tensor_90 = None
        copy__default_91: "f32[40]" = torch.ops.aten.copy_.default(arg191_1, add_tensor_91);  arg191_1 = add_tensor_91 = None
        copy__default_92: "f32[40]" = torch.ops.aten.copy_.default(arg192_1, add_tensor_92);  arg192_1 = add_tensor_92 = None
        copy__default_93: "i64[]" = torch.ops.aten.copy_.default(arg196_1, add_tensor_93);  arg196_1 = add_tensor_93 = None
        copy__default_94: "f32[40]" = torch.ops.aten.copy_.default(arg197_1, add_tensor_94);  arg197_1 = add_tensor_94 = None
        copy__default_95: "f32[40]" = torch.ops.aten.copy_.default(arg198_1, add_tensor_95);  arg198_1 = add_tensor_95 = None
        copy__default_96: "i64[]" = torch.ops.aten.copy_.default(arg202_1, add_tensor_96);  arg202_1 = add_tensor_96 = None
        copy__default_97: "f32[40]" = torch.ops.aten.copy_.default(arg203_1, add_tensor_97);  arg203_1 = add_tensor_97 = None
        copy__default_98: "f32[40]" = torch.ops.aten.copy_.default(arg204_1, add_tensor_98);  arg204_1 = add_tensor_98 = None
        copy__default_99: "i64[]" = torch.ops.aten.copy_.default(arg208_1, add_tensor_99);  arg208_1 = add_tensor_99 = None
        copy__default_100: "f32[80]" = torch.ops.aten.copy_.default(arg209_1, add_tensor_100);  arg209_1 = add_tensor_100 = None
        copy__default_101: "f32[80]" = torch.ops.aten.copy_.default(arg210_1, add_tensor_101);  arg210_1 = add_tensor_101 = None
        copy__default_102: "i64[]" = torch.ops.aten.copy_.default(arg214_1, add_tensor_102);  arg214_1 = add_tensor_102 = None
        copy__default_103: "f32[100]" = torch.ops.aten.copy_.default(arg215_1, add_tensor_103);  arg215_1 = add_tensor_103 = None
        copy__default_104: "f32[100]" = torch.ops.aten.copy_.default(arg216_1, add_tensor_104);  arg216_1 = add_tensor_104 = None
        copy__default_105: "i64[]" = torch.ops.aten.copy_.default(arg220_1, add_tensor_105);  arg220_1 = add_tensor_105 = None
        copy__default_106: "f32[100]" = torch.ops.aten.copy_.default(arg221_1, add_tensor_106);  arg221_1 = add_tensor_106 = None
        copy__default_107: "f32[100]" = torch.ops.aten.copy_.default(arg222_1, add_tensor_107);  arg222_1 = add_tensor_107 = None
        copy__default_108: "i64[]" = torch.ops.aten.copy_.default(arg226_1, add_tensor_108);  arg226_1 = add_tensor_108 = None
        copy__default_109: "f32[40]" = torch.ops.aten.copy_.default(arg227_1, add_tensor_109);  arg227_1 = add_tensor_109 = None
        copy__default_110: "f32[40]" = torch.ops.aten.copy_.default(arg228_1, add_tensor_110);  arg228_1 = add_tensor_110 = None
        copy__default_111: "i64[]" = torch.ops.aten.copy_.default(arg232_1, add_tensor_111);  arg232_1 = add_tensor_111 = None
        copy__default_112: "f32[40]" = torch.ops.aten.copy_.default(arg233_1, add_tensor_112);  arg233_1 = add_tensor_112 = None
        copy__default_113: "f32[40]" = torch.ops.aten.copy_.default(arg234_1, add_tensor_113);  arg234_1 = add_tensor_113 = None
        copy__default_114: "i64[]" = torch.ops.aten.copy_.default(arg238_1, add_tensor_114);  arg238_1 = add_tensor_114 = None
        copy__default_115: "f32[92]" = torch.ops.aten.copy_.default(arg239_1, add_tensor_115);  arg239_1 = add_tensor_115 = None
        copy__default_116: "f32[92]" = torch.ops.aten.copy_.default(arg240_1, add_tensor_116);  arg240_1 = add_tensor_116 = None
        copy__default_117: "i64[]" = torch.ops.aten.copy_.default(arg244_1, add_tensor_117);  arg244_1 = add_tensor_117 = None
        copy__default_118: "f32[92]" = torch.ops.aten.copy_.default(arg245_1, add_tensor_118);  arg245_1 = add_tensor_118 = None
        copy__default_119: "f32[92]" = torch.ops.aten.copy_.default(arg246_1, add_tensor_119);  arg246_1 = add_tensor_119 = None
        copy__default_120: "i64[]" = torch.ops.aten.copy_.default(arg250_1, add_tensor_120);  arg250_1 = add_tensor_120 = None
        copy__default_121: "f32[40]" = torch.ops.aten.copy_.default(arg251_1, add_tensor_121);  arg251_1 = add_tensor_121 = None
        copy__default_122: "f32[40]" = torch.ops.aten.copy_.default(arg252_1, add_tensor_122);  arg252_1 = add_tensor_122 = None
        copy__default_123: "i64[]" = torch.ops.aten.copy_.default(arg256_1, add_tensor_123);  arg256_1 = add_tensor_123 = None
        copy__default_124: "f32[40]" = torch.ops.aten.copy_.default(arg257_1, add_tensor_124);  arg257_1 = add_tensor_124 = None
        copy__default_125: "f32[40]" = torch.ops.aten.copy_.default(arg258_1, add_tensor_125);  arg258_1 = add_tensor_125 = None
        copy__default_126: "i64[]" = torch.ops.aten.copy_.default(arg262_1, add_tensor_126);  arg262_1 = add_tensor_126 = None
        copy__default_127: "f32[92]" = torch.ops.aten.copy_.default(arg263_1, add_tensor_127);  arg263_1 = add_tensor_127 = None
        copy__default_128: "f32[92]" = torch.ops.aten.copy_.default(arg264_1, add_tensor_128);  arg264_1 = add_tensor_128 = None
        copy__default_129: "i64[]" = torch.ops.aten.copy_.default(arg268_1, add_tensor_129);  arg268_1 = add_tensor_129 = None
        copy__default_130: "f32[92]" = torch.ops.aten.copy_.default(arg269_1, add_tensor_130);  arg269_1 = add_tensor_130 = None
        copy__default_131: "f32[92]" = torch.ops.aten.copy_.default(arg270_1, add_tensor_131);  arg270_1 = add_tensor_131 = None
        copy__default_132: "i64[]" = torch.ops.aten.copy_.default(arg274_1, add_tensor_132);  arg274_1 = add_tensor_132 = None
        copy__default_133: "f32[40]" = torch.ops.aten.copy_.default(arg275_1, add_tensor_133);  arg275_1 = add_tensor_133 = None
        copy__default_134: "f32[40]" = torch.ops.aten.copy_.default(arg276_1, add_tensor_134);  arg276_1 = add_tensor_134 = None
        copy__default_135: "i64[]" = torch.ops.aten.copy_.default(arg280_1, add_tensor_135);  arg280_1 = add_tensor_135 = None
        copy__default_136: "f32[40]" = torch.ops.aten.copy_.default(arg281_1, add_tensor_136);  arg281_1 = add_tensor_136 = None
        copy__default_137: "f32[40]" = torch.ops.aten.copy_.default(arg282_1, add_tensor_137);  arg282_1 = add_tensor_137 = None
        copy__default_138: "i64[]" = torch.ops.aten.copy_.default(arg286_1, add_tensor_138);  arg286_1 = add_tensor_138 = None
        copy__default_139: "f32[240]" = torch.ops.aten.copy_.default(arg287_1, add_tensor_139);  arg287_1 = add_tensor_139 = None
        copy__default_140: "f32[240]" = torch.ops.aten.copy_.default(arg288_1, add_tensor_140);  arg288_1 = add_tensor_140 = None
        copy__default_141: "i64[]" = torch.ops.aten.copy_.default(arg292_1, add_tensor_141);  arg292_1 = add_tensor_141 = None
        copy__default_142: "f32[240]" = torch.ops.aten.copy_.default(arg293_1, add_tensor_142);  arg293_1 = add_tensor_142 = None
        copy__default_143: "f32[240]" = torch.ops.aten.copy_.default(arg294_1, add_tensor_143);  arg294_1 = add_tensor_143 = None
        copy__default_144: "i64[]" = torch.ops.aten.copy_.default(arg302_1, add_tensor_144);  arg302_1 = add_tensor_144 = None
        copy__default_145: "f32[56]" = torch.ops.aten.copy_.default(arg303_1, add_tensor_145);  arg303_1 = add_tensor_145 = None
        copy__default_146: "f32[56]" = torch.ops.aten.copy_.default(arg304_1, add_tensor_146);  arg304_1 = add_tensor_146 = None
        copy__default_147: "i64[]" = torch.ops.aten.copy_.default(arg308_1, add_tensor_147);  arg308_1 = add_tensor_147 = None
        copy__default_148: "f32[56]" = torch.ops.aten.copy_.default(arg309_1, add_tensor_148);  arg309_1 = add_tensor_148 = None
        copy__default_149: "f32[56]" = torch.ops.aten.copy_.default(arg310_1, add_tensor_149);  arg310_1 = add_tensor_149 = None
        copy__default_150: "i64[]" = torch.ops.aten.copy_.default(arg314_1, add_tensor_150);  arg314_1 = add_tensor_150 = None
        copy__default_151: "f32[80]" = torch.ops.aten.copy_.default(arg315_1, add_tensor_151);  arg315_1 = add_tensor_151 = None
        copy__default_152: "f32[80]" = torch.ops.aten.copy_.default(arg316_1, add_tensor_152);  arg316_1 = add_tensor_152 = None
        copy__default_153: "i64[]" = torch.ops.aten.copy_.default(arg320_1, add_tensor_153);  arg320_1 = add_tensor_153 = None
        copy__default_154: "f32[112]" = torch.ops.aten.copy_.default(arg321_1, add_tensor_154);  arg321_1 = add_tensor_154 = None
        copy__default_155: "f32[112]" = torch.ops.aten.copy_.default(arg322_1, add_tensor_155);  arg322_1 = add_tensor_155 = None
        copy__default_156: "i64[]" = torch.ops.aten.copy_.default(arg326_1, add_tensor_156);  arg326_1 = add_tensor_156 = None
        copy__default_157: "f32[336]" = torch.ops.aten.copy_.default(arg327_1, add_tensor_157);  arg327_1 = add_tensor_157 = None
        copy__default_158: "f32[336]" = torch.ops.aten.copy_.default(arg328_1, add_tensor_158);  arg328_1 = add_tensor_158 = None
        copy__default_159: "i64[]" = torch.ops.aten.copy_.default(arg332_1, add_tensor_159);  arg332_1 = add_tensor_159 = None
        copy__default_160: "f32[336]" = torch.ops.aten.copy_.default(arg333_1, add_tensor_160);  arg333_1 = add_tensor_160 = None
        copy__default_161: "f32[336]" = torch.ops.aten.copy_.default(arg334_1, add_tensor_161);  arg334_1 = add_tensor_161 = None
        copy__default_162: "i64[]" = torch.ops.aten.copy_.default(arg342_1, add_tensor_162);  arg342_1 = add_tensor_162 = None
        copy__default_163: "f32[56]" = torch.ops.aten.copy_.default(arg343_1, add_tensor_163);  arg343_1 = add_tensor_163 = None
        copy__default_164: "f32[56]" = torch.ops.aten.copy_.default(arg344_1, add_tensor_164);  arg344_1 = add_tensor_164 = None
        copy__default_165: "i64[]" = torch.ops.aten.copy_.default(arg348_1, add_tensor_165);  arg348_1 = add_tensor_165 = None
        copy__default_166: "f32[56]" = torch.ops.aten.copy_.default(arg349_1, add_tensor_166);  arg349_1 = add_tensor_166 = None
        copy__default_167: "f32[56]" = torch.ops.aten.copy_.default(arg350_1, add_tensor_167);  arg350_1 = add_tensor_167 = None
        copy__default_168: "i64[]" = torch.ops.aten.copy_.default(arg354_1, add_tensor_168);  arg354_1 = add_tensor_168 = None
        copy__default_169: "f32[336]" = torch.ops.aten.copy_.default(arg355_1, add_tensor_169);  arg355_1 = add_tensor_169 = None
        copy__default_170: "f32[336]" = torch.ops.aten.copy_.default(arg356_1, add_tensor_170);  arg356_1 = add_tensor_170 = None
        copy__default_171: "i64[]" = torch.ops.aten.copy_.default(arg360_1, add_tensor_171);  arg360_1 = add_tensor_171 = None
        copy__default_172: "f32[336]" = torch.ops.aten.copy_.default(arg361_1, add_tensor_172);  arg361_1 = add_tensor_172 = None
        copy__default_173: "f32[336]" = torch.ops.aten.copy_.default(arg362_1, add_tensor_173);  arg362_1 = add_tensor_173 = None
        copy__default_174: "i64[]" = torch.ops.aten.copy_.default(arg366_1, add_tensor_174);  arg366_1 = add_tensor_174 = None
        copy__default_175: "f32[672]" = torch.ops.aten.copy_.default(arg367_1, add_tensor_175);  arg367_1 = add_tensor_175 = None
        copy__default_176: "f32[672]" = torch.ops.aten.copy_.default(arg368_1, add_tensor_176);  arg368_1 = add_tensor_176 = None
        copy__default_177: "i64[]" = torch.ops.aten.copy_.default(arg376_1, add_tensor_177);  arg376_1 = add_tensor_177 = None
        copy__default_178: "f32[80]" = torch.ops.aten.copy_.default(arg377_1, add_tensor_178);  arg377_1 = add_tensor_178 = None
        copy__default_179: "f32[80]" = torch.ops.aten.copy_.default(arg378_1, add_tensor_179);  arg378_1 = add_tensor_179 = None
        copy__default_180: "i64[]" = torch.ops.aten.copy_.default(arg382_1, add_tensor_180);  arg382_1 = add_tensor_180 = None
        copy__default_181: "f32[80]" = torch.ops.aten.copy_.default(arg383_1, add_tensor_181);  arg383_1 = add_tensor_181 = None
        copy__default_182: "f32[80]" = torch.ops.aten.copy_.default(arg384_1, add_tensor_182);  arg384_1 = add_tensor_182 = None
        copy__default_183: "i64[]" = torch.ops.aten.copy_.default(arg388_1, add_tensor_183);  arg388_1 = add_tensor_183 = None
        copy__default_184: "f32[112]" = torch.ops.aten.copy_.default(arg389_1, add_tensor_184);  arg389_1 = add_tensor_184 = None
        copy__default_185: "f32[112]" = torch.ops.aten.copy_.default(arg390_1, add_tensor_185);  arg390_1 = add_tensor_185 = None
        copy__default_186: "i64[]" = torch.ops.aten.copy_.default(arg394_1, add_tensor_186);  arg394_1 = add_tensor_186 = None
        copy__default_187: "f32[160]" = torch.ops.aten.copy_.default(arg395_1, add_tensor_187);  arg395_1 = add_tensor_187 = None
        copy__default_188: "f32[160]" = torch.ops.aten.copy_.default(arg396_1, add_tensor_188);  arg396_1 = add_tensor_188 = None
        copy__default_189: "i64[]" = torch.ops.aten.copy_.default(arg400_1, add_tensor_189);  arg400_1 = add_tensor_189 = None
        copy__default_190: "f32[480]" = torch.ops.aten.copy_.default(arg401_1, add_tensor_190);  arg401_1 = add_tensor_190 = None
        copy__default_191: "f32[480]" = torch.ops.aten.copy_.default(arg402_1, add_tensor_191);  arg402_1 = add_tensor_191 = None
        copy__default_192: "i64[]" = torch.ops.aten.copy_.default(arg406_1, add_tensor_192);  arg406_1 = add_tensor_192 = None
        copy__default_193: "f32[480]" = torch.ops.aten.copy_.default(arg407_1, add_tensor_193);  arg407_1 = add_tensor_193 = None
        copy__default_194: "f32[480]" = torch.ops.aten.copy_.default(arg408_1, add_tensor_194);  arg408_1 = add_tensor_194 = None
        copy__default_195: "i64[]" = torch.ops.aten.copy_.default(arg412_1, add_tensor_195);  arg412_1 = add_tensor_195 = None
        copy__default_196: "f32[80]" = torch.ops.aten.copy_.default(arg413_1, add_tensor_196);  arg413_1 = add_tensor_196 = None
        copy__default_197: "f32[80]" = torch.ops.aten.copy_.default(arg414_1, add_tensor_197);  arg414_1 = add_tensor_197 = None
        copy__default_198: "i64[]" = torch.ops.aten.copy_.default(arg418_1, add_tensor_198);  arg418_1 = add_tensor_198 = None
        copy__default_199: "f32[80]" = torch.ops.aten.copy_.default(arg419_1, add_tensor_199);  arg419_1 = add_tensor_199 = None
        copy__default_200: "f32[80]" = torch.ops.aten.copy_.default(arg420_1, add_tensor_200);  arg420_1 = add_tensor_200 = None
        copy__default_201: "i64[]" = torch.ops.aten.copy_.default(arg424_1, add_tensor_201);  arg424_1 = add_tensor_201 = None
        copy__default_202: "f32[480]" = torch.ops.aten.copy_.default(arg425_1, add_tensor_202);  arg425_1 = add_tensor_202 = None
        copy__default_203: "f32[480]" = torch.ops.aten.copy_.default(arg426_1, add_tensor_203);  arg426_1 = add_tensor_203 = None
        copy__default_204: "i64[]" = torch.ops.aten.copy_.default(arg430_1, add_tensor_204);  arg430_1 = add_tensor_204 = None
        copy__default_205: "f32[480]" = torch.ops.aten.copy_.default(arg431_1, add_tensor_205);  arg431_1 = add_tensor_205 = None
        copy__default_206: "f32[480]" = torch.ops.aten.copy_.default(arg432_1, add_tensor_206);  arg432_1 = add_tensor_206 = None
        copy__default_207: "i64[]" = torch.ops.aten.copy_.default(arg440_1, add_tensor_207);  arg440_1 = add_tensor_207 = None
        copy__default_208: "f32[80]" = torch.ops.aten.copy_.default(arg441_1, add_tensor_208);  arg441_1 = add_tensor_208 = None
        copy__default_209: "f32[80]" = torch.ops.aten.copy_.default(arg442_1, add_tensor_209);  arg442_1 = add_tensor_209 = None
        copy__default_210: "i64[]" = torch.ops.aten.copy_.default(arg446_1, add_tensor_210);  arg446_1 = add_tensor_210 = None
        copy__default_211: "f32[80]" = torch.ops.aten.copy_.default(arg447_1, add_tensor_211);  arg447_1 = add_tensor_211 = None
        copy__default_212: "f32[80]" = torch.ops.aten.copy_.default(arg448_1, add_tensor_212);  arg448_1 = add_tensor_212 = None
        copy__default_213: "i64[]" = torch.ops.aten.copy_.default(arg452_1, add_tensor_213);  arg452_1 = add_tensor_213 = None
        copy__default_214: "f32[480]" = torch.ops.aten.copy_.default(arg453_1, add_tensor_214);  arg453_1 = add_tensor_214 = None
        copy__default_215: "f32[480]" = torch.ops.aten.copy_.default(arg454_1, add_tensor_215);  arg454_1 = add_tensor_215 = None
        copy__default_216: "i64[]" = torch.ops.aten.copy_.default(arg458_1, add_tensor_216);  arg458_1 = add_tensor_216 = None
        copy__default_217: "f32[480]" = torch.ops.aten.copy_.default(arg459_1, add_tensor_217);  arg459_1 = add_tensor_217 = None
        copy__default_218: "f32[480]" = torch.ops.aten.copy_.default(arg460_1, add_tensor_218);  arg460_1 = add_tensor_218 = None
        copy__default_219: "i64[]" = torch.ops.aten.copy_.default(arg464_1, add_tensor_219);  arg464_1 = add_tensor_219 = None
        copy__default_220: "f32[80]" = torch.ops.aten.copy_.default(arg465_1, add_tensor_220);  arg465_1 = add_tensor_220 = None
        copy__default_221: "f32[80]" = torch.ops.aten.copy_.default(arg466_1, add_tensor_221);  arg466_1 = add_tensor_221 = None
        copy__default_222: "i64[]" = torch.ops.aten.copy_.default(arg470_1, add_tensor_222);  arg470_1 = add_tensor_222 = None
        copy__default_223: "f32[80]" = torch.ops.aten.copy_.default(arg471_1, add_tensor_223);  arg471_1 = add_tensor_223 = None
        copy__default_224: "f32[80]" = torch.ops.aten.copy_.default(arg472_1, add_tensor_224);  arg472_1 = add_tensor_224 = None
        copy__default_225: "i64[]" = torch.ops.aten.copy_.default(arg476_1, add_tensor_225);  arg476_1 = add_tensor_225 = None
        copy__default_226: "f32[480]" = torch.ops.aten.copy_.default(arg477_1, add_tensor_226);  arg477_1 = add_tensor_226 = None
        copy__default_227: "f32[480]" = torch.ops.aten.copy_.default(arg478_1, add_tensor_227);  arg478_1 = add_tensor_227 = None
        copy__default_228: "i64[]" = torch.ops.aten.copy_.default(arg482_1, add_tensor_228);  arg482_1 = add_tensor_228 = None
        copy__default_229: "f32[480]" = torch.ops.aten.copy_.default(arg483_1, add_tensor_229);  arg483_1 = add_tensor_229 = None
        copy__default_230: "f32[480]" = torch.ops.aten.copy_.default(arg484_1, add_tensor_230);  arg484_1 = add_tensor_230 = None
        copy__default_231: "i64[]" = torch.ops.aten.copy_.default(arg492_1, add_tensor_231);  arg492_1 = add_tensor_231 = None
        copy__default_232: "f32[80]" = torch.ops.aten.copy_.default(arg493_1, add_tensor_232);  arg493_1 = add_tensor_232 = None
        copy__default_233: "f32[80]" = torch.ops.aten.copy_.default(arg494_1, add_tensor_233);  arg494_1 = add_tensor_233 = None
        copy__default_234: "i64[]" = torch.ops.aten.copy_.default(arg498_1, add_tensor_234);  arg498_1 = add_tensor_234 = None
        copy__default_235: "f32[80]" = torch.ops.aten.copy_.default(arg499_1, add_tensor_235);  arg499_1 = add_tensor_235 = None
        copy__default_236: "f32[80]" = torch.ops.aten.copy_.default(arg500_1, add_tensor_236);  arg500_1 = add_tensor_236 = None
        copy__default_237: "i64[]" = torch.ops.aten.copy_.default(arg504_1, add_tensor_237);  arg504_1 = add_tensor_237 = None
        copy__default_238: "f32[960]" = torch.ops.aten.copy_.default(arg505_1, add_tensor_238);  arg505_1 = add_tensor_238 = None
        copy__default_239: "f32[960]" = torch.ops.aten.copy_.default(arg506_1, add_tensor_239);  arg506_1 = add_tensor_239 = None
        return (squeeze_dims_1, squeeze_dims_4, squeeze_dims_9, squeeze_dims_12, squeeze_dims_15, squeeze_dims_20, squeeze_dims_23, squeeze_dims_26, squeeze_dims_29, squeeze_dims_32, squeeze_dims_35, squeeze_dims_40, squeeze_dims_43, squeeze_dims_46, squeeze_dims_53, squeeze_dims_56, squeeze_dims_59, squeeze_dims_62, squeeze_dims_65, squeeze_dims_70, squeeze_dims_73, squeeze_dims_76, squeeze_dims_81, squeeze_dims_84, squeeze_dims_87, squeeze_dims_90, squeeze_dims_93, squeeze_dims_96, squeeze_dims_101, squeeze_dims_104, squeeze_dims_107, squeeze_dims_112, squeeze_dims_115, squeeze_dims_118, squeeze_dims_123, squeeze_dims_126, squeeze_dims_129, squeeze_dims_134, squeeze_dims_137, squeeze_dims_140, squeeze_dims_143, squeeze_dims_146, squeeze_dims_151, squeeze_dims_154, squeeze_dims_157, squeeze_dims_164, squeeze_dims_167, squeeze_dims_170, squeeze_dims_173, squeeze_dims_176, squeeze_dims_181, squeeze_dims_184, squeeze_dims_187, squeeze_dims_192, squeeze_dims_195, squeeze_dims_198, squeeze_dims_203, squeeze_dims_206, squeeze_dims_209, squeeze_dims_214, squeeze_dims_217, convert_element_type_default, view_default, permute_default_1, le_scalar, unsqueeze_default_2, unsqueeze_default_5, unsqueeze_default_8, unsqueeze_default_11, unsqueeze_default_14, unsqueeze_default_17, unsqueeze_default_20, unsqueeze_default_23, unsqueeze_default_26, unsqueeze_default_29, unsqueeze_default_32, unsqueeze_default_35, unsqueeze_default_38, unsqueeze_default_41, unsqueeze_default_44, unsqueeze_default_47, unsqueeze_default_50, unsqueeze_default_53, unsqueeze_default_56, unsqueeze_default_59, unsqueeze_default_62, unsqueeze_default_65, unsqueeze_default_68, unsqueeze_default_71, unsqueeze_default_74, unsqueeze_default_77, unsqueeze_default_80, unsqueeze_default_83, unsqueeze_default_86, unsqueeze_default_89, unsqueeze_default_92, unsqueeze_default_95, unsqueeze_default_98, unsqueeze_default_101, unsqueeze_default_104, unsqueeze_default_107, unsqueeze_default_110, unsqueeze_default_113, unsqueeze_default_116, unsqueeze_default_119, unsqueeze_default_122, unsqueeze_default_125, unsqueeze_default_128, unsqueeze_default_131, unsqueeze_default_134, unsqueeze_default_137, unsqueeze_default_140, unsqueeze_default_143, unsqueeze_default_146, unsqueeze_default_149, unsqueeze_default_152, unsqueeze_default_155, unsqueeze_default_158, unsqueeze_default_161, unsqueeze_default_164, unsqueeze_default_167, unsqueeze_default_170, unsqueeze_default_173, unsqueeze_default_176, unsqueeze_default_179, unsqueeze_default_182, copy__default, copy__default_1, copy__default_2, copy__default_3, copy__default_4, copy__default_5, copy__default_6, copy__default_7, copy__default_8, copy__default_9, copy__default_10, copy__default_11, copy__default_12, copy__default_13, copy__default_14, copy__default_15, copy__default_16, copy__default_17, copy__default_18, copy__default_19, copy__default_20, copy__default_21, copy__default_22, copy__default_23, copy__default_24, copy__default_25, copy__default_26, copy__default_27, copy__default_28, copy__default_29, copy__default_30, copy__default_31, copy__default_32, copy__default_33, copy__default_34, copy__default_35, copy__default_36, copy__default_37, copy__default_38, copy__default_39, copy__default_40, copy__default_41, copy__default_42, copy__default_43, copy__default_44, copy__default_45, copy__default_46, copy__default_47, copy__default_48, copy__default_49, copy__default_50, copy__default_51, copy__default_52, copy__default_53, copy__default_54, copy__default_55, copy__default_56, copy__default_57, copy__default_58, copy__default_59, copy__default_60, copy__default_61, copy__default_62, copy__default_63, copy__default_64, copy__default_65, copy__default_66, copy__default_67, copy__default_68, copy__default_69, copy__default_70, copy__default_71, copy__default_72, copy__default_73, copy__default_74, copy__default_75, copy__default_76, copy__default_77, copy__default_78, copy__default_79, copy__default_80, copy__default_81, copy__default_82, copy__default_83, copy__default_84, copy__default_85, copy__default_86, copy__default_87, copy__default_88, copy__default_89, copy__default_90, copy__default_91, copy__default_92, copy__default_93, copy__default_94, copy__default_95, copy__default_96, copy__default_97, copy__default_98, copy__default_99, copy__default_100, copy__default_101, copy__default_102, copy__default_103, copy__default_104, copy__default_105, copy__default_106, copy__default_107, copy__default_108, copy__default_109, copy__default_110, copy__default_111, copy__default_112, copy__default_113, copy__default_114, copy__default_115, copy__default_116, copy__default_117, copy__default_118, copy__default_119, copy__default_120, copy__default_121, copy__default_122, copy__default_123, copy__default_124, copy__default_125, copy__default_126, copy__default_127, copy__default_128, copy__default_129, copy__default_130, copy__default_131, copy__default_132, copy__default_133, copy__default_134, copy__default_135, copy__default_136, copy__default_137, copy__default_138, copy__default_139, copy__default_140, copy__default_141, copy__default_142, copy__default_143, copy__default_144, copy__default_145, copy__default_146, copy__default_147, copy__default_148, copy__default_149, copy__default_150, copy__default_151, copy__default_152, copy__default_153, copy__default_154, copy__default_155, copy__default_156, copy__default_157, copy__default_158, copy__default_159, copy__default_160, copy__default_161, copy__default_162, copy__default_163, copy__default_164, copy__default_165, copy__default_166, copy__default_167, copy__default_168, copy__default_169, copy__default_170, copy__default_171, copy__default_172, copy__default_173, copy__default_174, copy__default_175, copy__default_176, copy__default_177, copy__default_178, copy__default_179, copy__default_180, copy__default_181, copy__default_182, copy__default_183, copy__default_184, copy__default_185, copy__default_186, copy__default_187, copy__default_188, copy__default_189, copy__default_190, copy__default_191, copy__default_192, copy__default_193, copy__default_194, copy__default_195, copy__default_196, copy__default_197, copy__default_198, copy__default_199, copy__default_200, copy__default_201, copy__default_202, copy__default_203, copy__default_204, copy__default_205, copy__default_206, copy__default_207, copy__default_208, copy__default_209, copy__default_210, copy__default_211, copy__default_212, copy__default_213, copy__default_214, copy__default_215, copy__default_216, copy__default_217, copy__default_218, copy__default_219, copy__default_220, copy__default_221, copy__default_222, copy__default_223, copy__default_224, copy__default_225, copy__default_226, copy__default_227, copy__default_228, copy__default_229, copy__default_230, copy__default_231, copy__default_232, copy__default_233, copy__default_234, copy__default_235, copy__default_236, copy__default_237, copy__default_238, copy__default_239)


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
