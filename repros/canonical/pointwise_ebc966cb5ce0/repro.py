"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s5_g77
Pattern hash: ebc966cb5ce0
Shape hash: be2f6cb4
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem_70: "f32[]", getitem_71: "f32[]", getitem_72: "f32[]", getitem_73: "f32[]", getitem_74: "f32[]", getitem_75: "f32[]", getitem_76: "f32[]", getitem_77: "f32[]", getitem_78: "f32[]", getitem_79: "f32[]", getitem_130: "f32[16, 1]", getitem_131: "f32[16]", getitem_132: "f32[16, 16]", getitem_133: "f32[16]", getitem_134: "f32[16, 16]", getitem_135: "f32[16]", getitem_136: "f32[16, 16]", getitem_137: "f32[16]", getitem_138: "f32[1, 16]", getitem_139: "f32[1]", getitem_120: "f32[]", getitem_121: "f32[]", getitem_122: "f32[]", getitem_123: "f32[]", getitem_124: "f32[]", getitem_125: "f32[]", getitem_126: "f32[]", getitem_127: "f32[]", getitem_128: "f32[]", getitem_129: "f32[]"):
        # No stacktrace found for following nodes
        _foreach_div_scalar = torch.ops.aten._foreach_div.Scalar([getitem_70, getitem_71, getitem_72, getitem_73, getitem_74, getitem_75, getitem_76, getitem_77, getitem_78, getitem_79], 0.01);  getitem_70 = getitem_71 = getitem_72 = getitem_73 = getitem_74 = getitem_75 = getitem_76 = getitem_77 = getitem_78 = getitem_79 = None
        getitem: "f32[]" = _foreach_div_scalar[0]
        getitem_80: "f32[]" = _foreach_div_scalar[1]
        getitem_81: "f32[]" = _foreach_div_scalar[2]
        getitem_82: "f32[]" = _foreach_div_scalar[3]
        getitem_83: "f32[]" = _foreach_div_scalar[4]
        getitem_84: "f32[]" = _foreach_div_scalar[5]
        getitem_85: "f32[]" = _foreach_div_scalar[6]
        getitem_86: "f32[]" = _foreach_div_scalar[7]
        getitem_87: "f32[]" = _foreach_div_scalar[8]
        getitem_88: "f32[]" = _foreach_div_scalar[9];  _foreach_div_scalar = None
        _foreach_div_list = torch.ops.aten._foreach_div.List([getitem_130, getitem_131, getitem_132, getitem_133, getitem_134, getitem_135, getitem_136, getitem_137, getitem_138, getitem_139], [getitem_120, getitem_121, getitem_122, getitem_123, getitem_124, getitem_125, getitem_126, getitem_127, getitem_128, getitem_129]);  getitem_130 = getitem_131 = getitem_132 = getitem_133 = getitem_134 = getitem_135 = getitem_136 = getitem_137 = getitem_138 = getitem_139 = getitem_120 = getitem_121 = getitem_122 = getitem_123 = getitem_124 = getitem_125 = getitem_126 = getitem_127 = getitem_128 = getitem_129 = None
        getitem_140: "f32[16, 1]" = _foreach_div_list[0]
        getitem_141: "f32[16]" = _foreach_div_list[1]
        getitem_142: "f32[16, 16]" = _foreach_div_list[2]
        getitem_143: "f32[16]" = _foreach_div_list[3]
        getitem_144: "f32[16, 16]" = _foreach_div_list[4]
        getitem_145: "f32[16]" = _foreach_div_list[5]
        getitem_146: "f32[16, 16]" = _foreach_div_list[6]
        getitem_147: "f32[16]" = _foreach_div_list[7]
        getitem_148: "f32[1, 16]" = _foreach_div_list[8]
        getitem_149: "f32[1]" = _foreach_div_list[9];  _foreach_div_list = None
        return (getitem, getitem_80, getitem_81, getitem_82, getitem_83, getitem_84, getitem_85, getitem_86, getitem_87, getitem_88, getitem_140, getitem_141, getitem_142, getitem_143, getitem_144, getitem_145, getitem_146, getitem_147, getitem_148, getitem_149)


def _default_make_inputs():
    return [
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
