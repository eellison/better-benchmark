"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s5_g77
Pattern hash: a7780e0098a2
Shape hash: 4532b748
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg13_1: "f32[16, 1]", arg11_1: "f32[16]", arg24_1: "f32[16, 16]", arg25_1: "f32[16]", arg26_1: "f32[16, 16]", arg27_1: "f32[16]", arg28_1: "f32[16, 16]", arg29_1: "f32[16]", arg30_1: "f32[1, 16]", arg31_1: "f32[1]", getitem_10: "f32[16, 1]", getitem_11: "f32[16]", getitem_12: "f32[16, 16]", getitem_13: "f32[16]", getitem_14: "f32[16, 16]", getitem_15: "f32[16]", getitem_16: "f32[16, 16]", getitem_17: "f32[16]", getitem_18: "f32[1, 16]", getitem_19: "f32[1]", getitem_150: "f32[16, 1]", getitem_151: "f32[16]", getitem_152: "f32[16, 16]", getitem_153: "f32[16]", getitem_154: "f32[16, 16]", getitem_155: "f32[16]", getitem_156: "f32[16, 16]", getitem_157: "f32[16]", getitem_158: "f32[1, 16]", getitem_159: "f32[1]", getitem_110: "f32[]", getitem_111: "f32[]", getitem_112: "f32[]", getitem_113: "f32[]", getitem_114: "f32[]", getitem_115: "f32[]", getitem_116: "f32[]", getitem_117: "f32[]", getitem_118: "f32[]", getitem_119: "f32[]"):
        # No stacktrace found for following nodes
        full_default: "f32[16, 1]" = torch.ops.aten.full.default([16, 1], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "f32[16]" = torch.ops.aten.full.default([16], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_2: "f32[16, 16]" = torch.ops.aten.full.default([16, 16], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_3: "f32[16]" = torch.ops.aten.full.default([16], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_4: "f32[16, 16]" = torch.ops.aten.full.default([16, 16], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_5: "f32[16]" = torch.ops.aten.full.default([16], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_6: "f32[16, 16]" = torch.ops.aten.full.default([16, 16], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_7: "f32[16]" = torch.ops.aten.full.default([16], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_8: "f32[1, 16]" = torch.ops.aten.full.default([1, 16], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_9: "f32[1]" = torch.ops.aten.full.default([1], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _foreach_addcmul_scalar = torch.ops.aten._foreach_addcmul.Scalar([arg13_1, arg11_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1], [full_default, full_default_1, full_default_2, full_default_3, full_default_4, full_default_5, full_default_6, full_default_7, full_default_8, full_default_9], [getitem_10, getitem_11, getitem_12, getitem_13, getitem_14, getitem_15, getitem_16, getitem_17, getitem_18, getitem_19]);  arg13_1 = arg11_1 = arg24_1 = arg25_1 = arg26_1 = arg27_1 = arg28_1 = arg29_1 = arg30_1 = arg31_1 = full_default = full_default_1 = full_default_2 = full_default_3 = full_default_4 = full_default_5 = full_default_6 = full_default_7 = full_default_8 = full_default_9 = getitem_10 = getitem_11 = getitem_12 = getitem_13 = getitem_14 = getitem_15 = getitem_16 = getitem_17 = getitem_18 = getitem_19 = None
        getitem: "f32[16, 1]" = _foreach_addcmul_scalar[0]
        getitem_20: "f32[16]" = _foreach_addcmul_scalar[1]
        getitem_21: "f32[16, 16]" = _foreach_addcmul_scalar[2]
        getitem_22: "f32[16]" = _foreach_addcmul_scalar[3]
        getitem_23: "f32[16, 16]" = _foreach_addcmul_scalar[4]
        getitem_24: "f32[16]" = _foreach_addcmul_scalar[5]
        getitem_25: "f32[16, 16]" = _foreach_addcmul_scalar[6]
        getitem_26: "f32[16]" = _foreach_addcmul_scalar[7]
        getitem_27: "f32[1, 16]" = _foreach_addcmul_scalar[8]
        getitem_28: "f32[1]" = _foreach_addcmul_scalar[9];  _foreach_addcmul_scalar = None
        _foreach_div_list = torch.ops.aten._foreach_div.List([getitem_150, getitem_151, getitem_152, getitem_153, getitem_154, getitem_155, getitem_156, getitem_157, getitem_158, getitem_159], [getitem_110, getitem_111, getitem_112, getitem_113, getitem_114, getitem_115, getitem_116, getitem_117, getitem_118, getitem_119]);  getitem_150 = getitem_151 = getitem_152 = getitem_153 = getitem_154 = getitem_155 = getitem_156 = getitem_157 = getitem_158 = getitem_159 = getitem_110 = getitem_111 = getitem_112 = getitem_113 = getitem_114 = getitem_115 = getitem_116 = getitem_117 = getitem_118 = getitem_119 = None
        getitem_120: "f32[16, 1]" = _foreach_div_list[0]
        getitem_121: "f32[16]" = _foreach_div_list[1]
        getitem_122: "f32[16, 16]" = _foreach_div_list[2]
        getitem_123: "f32[16]" = _foreach_div_list[3]
        getitem_124: "f32[16, 16]" = _foreach_div_list[4]
        getitem_125: "f32[16]" = _foreach_div_list[5]
        getitem_126: "f32[16, 16]" = _foreach_div_list[6]
        getitem_127: "f32[16]" = _foreach_div_list[7]
        getitem_128: "f32[1, 16]" = _foreach_div_list[8]
        getitem_129: "f32[1]" = _foreach_div_list[9];  _foreach_div_list = None
        return (getitem, getitem_20, getitem_21, getitem_22, getitem_23, getitem_24, getitem_25, getitem_26, getitem_27, getitem_28, getitem_120, getitem_121, getitem_122, getitem_123, getitem_124, getitem_125, getitem_126, getitem_127, getitem_128, getitem_129)


def _default_make_inputs():
    return [
    torch.randn([16, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16], dtype=torch.float32, device='cuda'),
    torch.randn([16, 16], dtype=torch.float32, device='cuda'),
    torch.randn([16], dtype=torch.float32, device='cuda'),
    torch.randn([16, 16], dtype=torch.float32, device='cuda'),
    torch.randn([16], dtype=torch.float32, device='cuda'),
    torch.randn([16, 16], dtype=torch.float32, device='cuda'),
    torch.randn([16], dtype=torch.float32, device='cuda'),
    torch.randn([1, 16], dtype=torch.float32, device='cuda'),
    torch.randn([1], dtype=torch.float32, device='cuda'),
    torch.randn([16, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16], dtype=torch.float32, device='cuda'),
    torch.randn([16, 16], dtype=torch.float32, device='cuda'),
    torch.randn([16], dtype=torch.float32, device='cuda'),
    torch.randn([16, 16], dtype=torch.float32, device='cuda'),
    torch.randn([16], dtype=torch.float32, device='cuda'),
    torch.randn([16, 16], dtype=torch.float32, device='cuda'),
    torch.randn([16], dtype=torch.float32, device='cuda'),
    torch.randn([1, 16], dtype=torch.float32, device='cuda'),
    torch.randn([1], dtype=torch.float32, device='cuda'),
    torch.randn([16, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16], dtype=torch.float32, device='cuda'),
    torch.randn([16, 16], dtype=torch.float32, device='cuda'),
    torch.randn([16], dtype=torch.float32, device='cuda'),
    torch.randn([16, 16], dtype=torch.float32, device='cuda'),
    torch.randn([16], dtype=torch.float32, device='cuda'),
    torch.randn([16, 16], dtype=torch.float32, device='cuda'),
    torch.randn([16], dtype=torch.float32, device='cuda'),
    torch.randn([1, 16], dtype=torch.float32, device='cuda'),
    torch.randn([1], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
