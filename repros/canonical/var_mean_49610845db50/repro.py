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
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, addmm_69: "f16[2048, 768]", view_155: "f16[4, 512, 768]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:191 in forward, code: attn_output = self.out_proj(attn_output)
        reshape_default: "f16[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_69, _shape_param_0);  addmm_69 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:266 in forward, code: hidden_states = residual + hidden_states
        add_tensor: "f16[4, 512, 768]" = torch.ops.aten.add.Tensor(view_155, reshape_default);  view_155 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:274 in forward, code: hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        reshape_default_1: "f16[2048, 768]" = torch.ops.aten.reshape.default(add_tensor, _shape_param_1);  add_tensor = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:279 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_default: "f32[2048, 768]" = torch.ops.prims.convert_element_type.default(reshape_default_1, torch.float32);  reshape_default_1 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [1], correction = 0, keepdim = True);  convert_element_type_default = None
        getitem: "f32[2048, 1]" = var_mean_correction[0]
        getitem_1: "f32[2048, 1]" = var_mean_correction[1];  var_mean_correction = None
        _output_to_half_0: "f16[2048, 1]" = torch.ops.prims.convert_element_type.default(getitem, torch.float16);  getitem = None
        _output_to_half_1: "f16[2048, 1]" = torch.ops.prims.convert_element_type.default(getitem_1, torch.float16);  getitem_1 = None
        return (_output_to_half_0, _output_to_half_1)


def _default_make_inputs():
    return [
    torch.randn([2048, 768], dtype=torch.float16, device='cuda'),
    torch.randn([4, 512, 768], dtype=torch.float16, device='cuda'),
    [4, 512, 768],  # _shape_param_0
    [-1, 768],  # _shape_param_1
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
