"""
Standalone repro captured via capture_hook.
Label: repros/models/timm/train/vit_base_patch16_siglip_256/full_graph_001.py
Pattern hash: ffc423127a11
Shape hash: 47413718
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 3
# Input shapes/strides/dtypes live in the sibling shapes.json (structured,
# one entry per point); forward()'s annotations document the default shapes
# inline. Default inputs = the first shapes.json point.

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[128, 768]", arg1_1: "bf16[128, 3072]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        full: "bf16[128, 1, 768]" = torch.ops.aten.full.default(_shape_param_0, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_0 = None
        select_scatter: "bf16[128, 1, 768]" = torch.ops.aten.select_scatter.default(full, arg0_1, 1, 0);  full = arg0_1 = None
        view: "bf16[128, 768]" = torch.ops.aten.view.default(select_scatter, _shape_param_1);  select_scatter = _shape_param_1 = None
        permute: "bf16[768, 128]" = torch.ops.aten.permute.default(view, [1, 0]);  view = None
        mm: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute, arg1_1);  permute = arg1_1 = None
        convert_element_type: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        return convert_element_type



def _default_make_inputs():
    configs = load_shape_configs(__file__)
    if not configs:
        raise RuntimeError(
            "no shapes.json next to this repro — pass an explicit config "
            "via make_inputs(shape_config=...)")
    return make_inputs_from_config(next(iter(configs.values())))


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
