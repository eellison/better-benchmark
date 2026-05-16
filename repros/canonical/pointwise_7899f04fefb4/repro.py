"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s1_g244
Pattern hash: 7899f04fefb4
Shape hash: 84120312
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg9_1: "f32[64]", arg7_1: "f32[64, 64]", arg16_1: "f32[64]", arg17_1: "f32[64, 64]", arg18_1: "f32[64]", arg19_1: "f32[64, 64]", getitem_6: "f32[64]", getitem_7: "f32[64, 64]", getitem_8: "f32[64]", getitem_9: "f32[64, 64]", getitem_10: "f32[64]", getitem_11: "f32[64, 64]", getitem_90: "f32[64]", getitem_91: "f32[64, 64]", getitem_92: "f32[64]", getitem_93: "f32[64, 64]", getitem_94: "f32[64]", getitem_95: "f32[64, 64]", getitem_66: "f32[]", getitem_67: "f32[]", getitem_68: "f32[]", getitem_69: "f32[]", getitem_70: "f32[]", getitem_71: "f32[]"):
        # No stacktrace found for following nodes
        full_default: "f32[64]" = torch.ops.aten.full.default([64], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "f32[64, 64]" = torch.ops.aten.full.default([64, 64], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_2: "f32[64]" = torch.ops.aten.full.default([64], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_3: "f32[64, 64]" = torch.ops.aten.full.default([64, 64], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_4: "f32[64]" = torch.ops.aten.full.default([64], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_5: "f32[64, 64]" = torch.ops.aten.full.default([64, 64], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _foreach_addcmul_scalar = torch.ops.aten._foreach_addcmul.Scalar([arg9_1, arg7_1, arg16_1, arg17_1, arg18_1, arg19_1], [full_default, full_default_1, full_default_2, full_default_3, full_default_4, full_default_5], [getitem_6, getitem_7, getitem_8, getitem_9, getitem_10, getitem_11]);  arg9_1 = arg7_1 = arg16_1 = arg17_1 = arg18_1 = arg19_1 = full_default = full_default_1 = full_default_2 = full_default_3 = full_default_4 = full_default_5 = getitem_6 = getitem_7 = getitem_8 = getitem_9 = getitem_10 = getitem_11 = None
        getitem: "f32[64]" = _foreach_addcmul_scalar[0]
        getitem_12: "f32[64, 64]" = _foreach_addcmul_scalar[1]
        getitem_13: "f32[64]" = _foreach_addcmul_scalar[2]
        getitem_14: "f32[64, 64]" = _foreach_addcmul_scalar[3]
        getitem_15: "f32[64]" = _foreach_addcmul_scalar[4]
        getitem_16: "f32[64, 64]" = _foreach_addcmul_scalar[5];  _foreach_addcmul_scalar = None
        _foreach_div_list = torch.ops.aten._foreach_div.List([getitem_90, getitem_91, getitem_92, getitem_93, getitem_94, getitem_95], [getitem_66, getitem_67, getitem_68, getitem_69, getitem_70, getitem_71]);  getitem_90 = getitem_91 = getitem_92 = getitem_93 = getitem_94 = getitem_95 = getitem_66 = getitem_67 = getitem_68 = getitem_69 = getitem_70 = getitem_71 = None
        getitem_72: "f32[64]" = _foreach_div_list[0]
        getitem_73: "f32[64, 64]" = _foreach_div_list[1]
        getitem_74: "f32[64]" = _foreach_div_list[2]
        getitem_75: "f32[64, 64]" = _foreach_div_list[3]
        getitem_76: "f32[64]" = _foreach_div_list[4]
        getitem_77: "f32[64, 64]" = _foreach_div_list[5];  _foreach_div_list = None
        return (getitem, getitem_12, getitem_13, getitem_14, getitem_15, getitem_16, getitem_72, getitem_73, getitem_74, getitem_75, getitem_76, getitem_77)


def _default_make_inputs():
    return [
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 64], dtype=torch.float32, device='cuda'),
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
