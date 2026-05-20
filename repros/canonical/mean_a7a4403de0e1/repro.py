"""
Standalone repro captured via capture_hook.
Label: timm_dm_nfnet_f0_infer
Pattern hash: a7a4403de0e1
Shape hash: 1e1527f6
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([128, 3072, 6, 6], f32, stride=(110592, 1, 18432, 3072)), T([1000, 3072], f32), S([128, 3072]))"

class Repro(torch.nn.Module):
    def forward(self, convolution_80: "f32[128, 3072, 6, 6]", arg232_1: "f32[1000, 3072]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        mul_tensor: "f32[128, 3072, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_80, 0.5)
        mul_tensor_1: "f32[128, 3072, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_80, 0.7071067811865476);  convolution_80 = None
        erf_default: "f32[128, 3072, 6, 6]" = torch.ops.aten.erf.default(mul_tensor_1);  mul_tensor_1 = None
        add_tensor: "f32[128, 3072, 6, 6]" = torch.ops.aten.add.Tensor(erf_default, 1);  erf_default = None
        mul_tensor_2: "f32[128, 3072, 6, 6]" = torch.ops.aten.mul.Tensor(mul_tensor, add_tensor);  mul_tensor = add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:89 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_tensor_3: "f32[128, 3072, 6, 6]" = torch.ops.aten.mul.Tensor(mul_tensor_2, 1.7015043497085571);  mul_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        mean_dim: "f32[128, 3072, 1, 1]" = torch.ops.aten.mean.dim(mul_tensor_3, [-1, -2], True);  mul_tensor_3 = None
        as_strided_default: "f32[128, 3072, 1, 1]" = torch.ops.aten.as_strided.default(mean_dim, [128, 3072, 1, 1], [3072, 1, 3072, 3072]);  mean_dim = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:173 in forward, code: x = self.flatten(x)
        reshape_default: "f32[128, 3072]" = torch.ops.aten.reshape.default(as_strided_default, _shape_param_0);  as_strided_default = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/classifier.py:141 in forward, code: x = self.fc(x)
        permute_default: "f32[3072, 1000]" = torch.ops.aten.permute.default(arg232_1, [1, 0]);  arg232_1 = None
        return (reshape_default, permute_default)


def _default_make_inputs():
    from repro_harness import parse_shapes_config
    return parse_shapes_config(_shapes_config)


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
