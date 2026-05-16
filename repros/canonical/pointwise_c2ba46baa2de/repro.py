"""
Standalone repro captured via capture_hook.
Label: timm_dm_nfnet_f0_training
Pattern hash: c2ba46baa2de
Shape hash: 465cf230
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm: "f32[32, 3072]", _shape_param_0, _shape_param_1, convolution_80: "f32[32, 3072, 6, 6]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:173 in forward, code: x = self.flatten(x)
        reshape_default: "f32[32, 3072, 1, 1]" = torch.ops.aten.reshape.default(mm, _shape_param_0);  mm = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        expand_default: "f32[32, 3072, 6, 6]" = torch.ops.aten.expand.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        div_scalar: "f32[32, 3072, 6, 6]" = torch.ops.aten.div.Scalar(expand_default, 36);  expand_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:89 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_tensor: "f32[32, 3072, 6, 6]" = torch.ops.aten.mul.Tensor(div_scalar, 1.7015043497085571);  div_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        mul_tensor_1: "f32[32, 3072, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_80, 0.7071067811865476)
        erf_default: "f32[32, 3072, 6, 6]" = torch.ops.aten.erf.default(mul_tensor_1);  mul_tensor_1 = None
        add_tensor: "f32[32, 3072, 6, 6]" = torch.ops.aten.add.Tensor(erf_default, 1);  erf_default = None
        mul_tensor_2: "f32[32, 3072, 6, 6]" = torch.ops.aten.mul.Tensor(add_tensor, 0.5);  add_tensor = None
        mul_tensor_3: "f32[32, 3072, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_80, convolution_80)
        mul_tensor_4: "f32[32, 3072, 6, 6]" = torch.ops.aten.mul.Tensor(mul_tensor_3, -0.5);  mul_tensor_3 = None
        exp_default: "f32[32, 3072, 6, 6]" = torch.ops.aten.exp.default(mul_tensor_4);  mul_tensor_4 = None
        mul_tensor_5: "f32[32, 3072, 6, 6]" = torch.ops.aten.mul.Tensor(exp_default, 0.3989422804014327);  exp_default = None
        mul_tensor_6: "f32[32, 3072, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_80, mul_tensor_5);  convolution_80 = mul_tensor_5 = None
        add_tensor_1: "f32[32, 3072, 6, 6]" = torch.ops.aten.add.Tensor(mul_tensor_2, mul_tensor_6);  mul_tensor_2 = mul_tensor_6 = None
        mul_tensor_7: "f32[32, 3072, 6, 6]" = torch.ops.aten.mul.Tensor(mul_tensor, add_tensor_1);  mul_tensor = add_tensor_1 = None
        return mul_tensor_7


def _default_make_inputs():
    return [
    torch.randn([32, 3072], dtype=torch.float32, device='cuda'),
    [32, 3072, 1, 1],  # _shape_param_0
    [32, 3072, 6, 6],  # _shape_param_1
    torch.randn([32, 3072, 6, 6], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
