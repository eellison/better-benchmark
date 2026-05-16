"""
Standalone repro captured via capture_hook.
Label: timm_nfnet_l0_inference
Pattern hash: 72234410cca9
Shape hash: db3b0772
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, convolution_80: "f32[32, 2304, 9, 9]", _shape_param_0, arg220_1: "f32[1000, 2304]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:569 in forward_features, code: x = self.final_act(x)
        neg_default: "f32[32, 2304, 9, 9]" = torch.ops.aten.neg.default(convolution_80)
        exp_default: "f32[32, 2304, 9, 9]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor: "f32[32, 2304, 9, 9]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        div_tensor: "f32[32, 2304, 9, 9]" = torch.ops.aten.div.Tensor(convolution_80, add_tensor);  convolution_80 = add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        mean_dim: "f32[32, 2304, 1, 1]" = torch.ops.aten.mean.dim(div_tensor, [-1, -2], True);  div_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:173 in forward, code: x = self.flatten(x)
        reshape_default: "f32[32, 2304]" = torch.ops.aten.reshape.default(mean_dim, _shape_param_0);  mean_dim = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/classifier.py:141 in forward, code: x = self.fc(x)
        permute_default: "f32[2304, 1000]" = torch.ops.aten.permute.default(arg220_1, [1, 0]);  arg220_1 = None
        return (reshape_default, permute_default)


def _default_make_inputs():
    return [
    torch.randn([32, 2304, 9, 9], dtype=torch.float32, device='cuda'),
    [32, 2304],  # _shape_param_0
    torch.randn([1000, 2304], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
