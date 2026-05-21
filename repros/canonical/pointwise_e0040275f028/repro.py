"""
Standalone repro captured via capture_hook.
Label: torchbench_moco_infer
Pattern hash: e0040275f028
Shape hash: d0baddd1
"""
import sys
from pathlib import Path

import sys
from pathlib import Path
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, bmm: "f32[32, 1, 1]", mm_default: "f32[32, 32000]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/utils/_device.py:122 in __torch_function__, code: return func(*args, **kwargs)
        full_default: "f32[32, 128]" = torch.ops.aten.full.default([32, 128], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        reshape_default: "f32[32, 1]" = torch.ops.aten.reshape.default(bmm, _shape_param_0);  bmm = _shape_param_0 = None
        permute_default: "f32[32, 1]" = torch.ops.aten.permute.default(reshape_default, [0, 1]);  reshape_default = None
        reshape_default_1: "f32[32]" = torch.ops.aten.reshape.default(permute_default, _shape_param_1);  permute_default = _shape_param_1 = None
        unsqueeze_default: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(reshape_default_1, -1);  reshape_default_1 = None
        unsqueeze_default_1: "f32[1, 32, 32000]" = torch.ops.aten.unsqueeze.default(mm_default, 0);  mm_default = None
        reshape_default_2: "f32[32, 1, 32000]" = torch.ops.aten.reshape.default(unsqueeze_default_1, _shape_param_2);  unsqueeze_default_1 = _shape_param_2 = None
        permute_default_1: "f32[32, 32000, 1]" = torch.ops.aten.permute.default(reshape_default_2, [0, 2, 1]);  reshape_default_2 = None
        reshape_default_3: "f32[32, 32000]" = torch.ops.aten.reshape.default(permute_default_1, _shape_param_3);  permute_default_1 = _shape_param_3 = None
        cat_default: "f32[32, 32001]" = torch.ops.aten.cat.default([unsqueeze_default, reshape_default_3], 1);  unsqueeze_default = reshape_default_3 = None
        div_tensor: "f32[32, 32001]" = torch.ops.aten.div.Tensor(cat_default, 0.07);  cat_default = None
        full_default_1: "i64[32]" = torch.ops.aten.full.default([32], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        return (full_default, div_tensor, full_default_1)


def _default_make_inputs():
    return []


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
