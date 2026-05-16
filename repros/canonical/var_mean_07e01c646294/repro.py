"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=var_mean, ranges=[], reduction_ranges=[]
#   origins: ['aten.var_mean.correction']
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, view_168: "f16[2048, 768]", addmm_71: "f16[2048, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:287 in forward, code: hidden_states = (residual + hidden_states).view(hidden_states_shape)
        add_tensor: "f16[2048, 768]" = torch.ops.aten.add.Tensor(view_168, addmm_71);  view_168 = addmm_71 = None
        reshape_default: "f16[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor, [4, 512, 768]);  add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:666 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_default: "f32[4, 512, 768]" = torch.ops.prims.convert_element_type.default(reshape_default, torch.float32);  reshape_default = None
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [2], correction = 0, keepdim = True);  convert_element_type_default = None
        getitem: "f32[4, 512, 1]" = var_mean_correction[0]
        getitem_1: "f32[4, 512, 1]" = var_mean_correction[1];  var_mean_correction = None
        return (getitem, getitem_1)


def _default_make_inputs():
    return [
    torch.randn([2048, 768], dtype=torch.float16, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
