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
    def forward(self, addmm_69: "f32[8192, 768]", view_155: "f32[4, 2048, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:191 in forward, code: attn_output = self.out_proj(attn_output)
        reshape_default: "f32[4, 2048, 768]" = torch.ops.aten.reshape.default(addmm_69, [4, 2048, 768]);  addmm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:266 in forward, code: hidden_states = residual + hidden_states
        add_tensor: "f32[4, 2048, 768]" = torch.ops.aten.add.Tensor(view_155, reshape_default);  view_155 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:274 in forward, code: hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        reshape_default_1: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_tensor, [-1, 768]);  add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:279 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_correction = torch.ops.aten.var_mean.correction(reshape_default_1, [1], correction = 0, keepdim = True);  reshape_default_1 = None
        getitem: "f32[8192, 1]" = var_mean_correction[0]
        getitem_1: "f32[8192, 1]" = var_mean_correction[1];  var_mean_correction = None
        return (getitem, getitem_1)


def _default_make_inputs():
    return [
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 2048, 768], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
