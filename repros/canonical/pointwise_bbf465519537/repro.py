"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s5_g77
Pattern hash: bbf465519537
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
    def forward(self, arg40_1: "f32[16, 1]", arg41_1: "f32[16]", arg42_1: "f32[16, 16]", arg43_1: "f32[16]", arg44_1: "f32[16, 16]", arg45_1: "f32[16]", arg46_1: "f32[16, 16]", arg47_1: "f32[16]", arg48_1: "f32[1, 16]", arg49_1: "f32[1]", arg13_1: "f32[16, 1]", arg11_1: "f32[16]", arg24_1: "f32[16, 16]", arg25_1: "f32[16]", arg26_1: "f32[16, 16]", arg27_1: "f32[16]", arg28_1: "f32[16, 16]", arg29_1: "f32[16]", arg30_1: "f32[1, 16]", arg31_1: "f32[1]", getitem_100: "f32[]", getitem_101: "f32[]", getitem_102: "f32[]", getitem_103: "f32[]", getitem_104: "f32[]", getitem_105: "f32[]", getitem_106: "f32[]", getitem_107: "f32[]", getitem_108: "f32[]", getitem_109: "f32[]", getitem_140: "f32[16, 1]", getitem_141: "f32[16]", getitem_142: "f32[16, 16]", getitem_143: "f32[16]", getitem_144: "f32[16, 16]", getitem_145: "f32[16]", getitem_146: "f32[16, 16]", getitem_147: "f32[16]", getitem_148: "f32[1, 16]", getitem_149: "f32[1]"):
        # No stacktrace found for following nodes
        _foreach_sub_list = torch.ops.aten._foreach_sub.List([arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1], [arg13_1, arg11_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1]);  arg40_1 = arg41_1 = arg42_1 = arg43_1 = arg44_1 = arg45_1 = arg46_1 = arg47_1 = arg48_1 = arg49_1 = arg13_1 = arg11_1 = arg24_1 = arg25_1 = arg26_1 = arg27_1 = arg28_1 = arg29_1 = arg30_1 = arg31_1 = None
        getitem: "f32[16, 1]" = _foreach_sub_list[0]
        getitem_1: "f32[16]" = _foreach_sub_list[1]
        getitem_2: "f32[16, 16]" = _foreach_sub_list[2]
        getitem_3: "f32[16]" = _foreach_sub_list[3]
        getitem_4: "f32[16, 16]" = _foreach_sub_list[4]
        getitem_5: "f32[16]" = _foreach_sub_list[5]
        getitem_6: "f32[16, 16]" = _foreach_sub_list[6]
        getitem_7: "f32[16]" = _foreach_sub_list[7]
        getitem_8: "f32[1, 16]" = _foreach_sub_list[8]
        getitem_9: "f32[1]" = _foreach_sub_list[9];  _foreach_sub_list = None
        _foreach_reciprocal_default = torch.ops.aten._foreach_reciprocal.default([getitem_100, getitem_101, getitem_102, getitem_103, getitem_104, getitem_105, getitem_106, getitem_107, getitem_108, getitem_109]);  getitem_100 = getitem_101 = getitem_102 = getitem_103 = getitem_104 = getitem_105 = getitem_106 = getitem_107 = getitem_108 = getitem_109 = None
        getitem_110: "f32[]" = _foreach_reciprocal_default[0]
        getitem_111: "f32[]" = _foreach_reciprocal_default[1]
        getitem_112: "f32[]" = _foreach_reciprocal_default[2]
        getitem_113: "f32[]" = _foreach_reciprocal_default[3]
        getitem_114: "f32[]" = _foreach_reciprocal_default[4]
        getitem_115: "f32[]" = _foreach_reciprocal_default[5]
        getitem_116: "f32[]" = _foreach_reciprocal_default[6]
        getitem_117: "f32[]" = _foreach_reciprocal_default[7]
        getitem_118: "f32[]" = _foreach_reciprocal_default[8]
        getitem_119: "f32[]" = _foreach_reciprocal_default[9];  _foreach_reciprocal_default = None
        _foreach_add_scalar = torch.ops.aten._foreach_add.Scalar([getitem_140, getitem_141, getitem_142, getitem_143, getitem_144, getitem_145, getitem_146, getitem_147, getitem_148, getitem_149], 1e-08);  getitem_140 = getitem_141 = getitem_142 = getitem_143 = getitem_144 = getitem_145 = getitem_146 = getitem_147 = getitem_148 = getitem_149 = None
        getitem_150: "f32[16, 1]" = _foreach_add_scalar[0]
        getitem_151: "f32[16]" = _foreach_add_scalar[1]
        getitem_152: "f32[16, 16]" = _foreach_add_scalar[2]
        getitem_153: "f32[16]" = _foreach_add_scalar[3]
        getitem_154: "f32[16, 16]" = _foreach_add_scalar[4]
        getitem_155: "f32[16]" = _foreach_add_scalar[5]
        getitem_156: "f32[16, 16]" = _foreach_add_scalar[6]
        getitem_157: "f32[16]" = _foreach_add_scalar[7]
        getitem_158: "f32[1, 16]" = _foreach_add_scalar[8]
        getitem_159: "f32[1]" = _foreach_add_scalar[9];  _foreach_add_scalar = None
        return (getitem, getitem_1, getitem_2, getitem_3, getitem_4, getitem_5, getitem_6, getitem_7, getitem_8, getitem_9, getitem_110, getitem_111, getitem_112, getitem_113, getitem_114, getitem_115, getitem_116, getitem_117, getitem_118, getitem_119, getitem_150, getitem_151, getitem_152, getitem_153, getitem_154, getitem_155, getitem_156, getitem_157, getitem_158, getitem_159)


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
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
