"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s1_g244
Pattern hash: 600636667ef5
Shape hash: 84c9dc21
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem_42: "f32[]", getitem_43: "f32[]", getitem_44: "f32[]", getitem_45: "f32[]", getitem_46: "f32[]", getitem_47: "f32[]", getitem_78: "f32[64]", getitem_79: "f32[64, 64]", getitem_80: "f32[64]", getitem_81: "f32[64, 64]", getitem_82: "f32[64]", getitem_83: "f32[64, 64]", getitem_72: "f32[]", getitem_73: "f32[]", getitem_74: "f32[]", getitem_75: "f32[]", getitem_76: "f32[]", getitem_77: "f32[]"):
        # No stacktrace found for following nodes
        _foreach_div_scalar = torch.ops.aten._foreach_div.Scalar([getitem_42, getitem_43, getitem_44, getitem_45, getitem_46, getitem_47], 0.01);  getitem_42 = getitem_43 = getitem_44 = getitem_45 = getitem_46 = getitem_47 = None
        getitem: "f32[]" = _foreach_div_scalar[0]
        getitem_48: "f32[]" = _foreach_div_scalar[1]
        getitem_49: "f32[]" = _foreach_div_scalar[2]
        getitem_50: "f32[]" = _foreach_div_scalar[3]
        getitem_51: "f32[]" = _foreach_div_scalar[4]
        getitem_52: "f32[]" = _foreach_div_scalar[5];  _foreach_div_scalar = None
        _foreach_div_list = torch.ops.aten._foreach_div.List([getitem_78, getitem_79, getitem_80, getitem_81, getitem_82, getitem_83], [getitem_72, getitem_73, getitem_74, getitem_75, getitem_76, getitem_77]);  getitem_78 = getitem_79 = getitem_80 = getitem_81 = getitem_82 = getitem_83 = getitem_72 = getitem_73 = getitem_74 = getitem_75 = getitem_76 = getitem_77 = None
        getitem_84: "f32[64]" = _foreach_div_list[0]
        getitem_85: "f32[64, 64]" = _foreach_div_list[1]
        getitem_86: "f32[64]" = _foreach_div_list[2]
        getitem_87: "f32[64, 64]" = _foreach_div_list[3]
        getitem_88: "f32[64]" = _foreach_div_list[4]
        getitem_89: "f32[64, 64]" = _foreach_div_list[5];  _foreach_div_list = None
        return (getitem, getitem_48, getitem_49, getitem_50, getitem_51, getitem_52, getitem_84, getitem_85, getitem_86, getitem_87, getitem_88, getitem_89)


def _default_make_inputs():
    return [
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 64], dtype=torch.float32, device='cuda'),
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
