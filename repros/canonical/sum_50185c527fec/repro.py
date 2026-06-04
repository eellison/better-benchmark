"""
Standalone repro captured via capture_hook.
Label: timm_swin_base_patch4_window7_224_train
Pattern hash: 50185c527fec
Shape hash: 092f66b3
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([32768, 49, 49], f32), T([8192, 4, 49, 49], f32, stride=(9728, 2432, 49, 1)), S([8192, 4, 49, 49]), S([32768, 49, 49]))"

class Repro(torch.nn.Module):
    def forward(self, bmm_137: "f32[32768, 49, 49]", div_1: "f32[8192, 4, 49, 49]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        reshape_default: "f32[8192, 4, 49, 49]" = torch.ops.aten.reshape.default(bmm_137, _shape_param_0);  bmm_137 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        mul_tensor: "f32[8192, 4, 49, 49]" = torch.ops.aten.mul.Tensor(reshape_default, div_1);  reshape_default = None
        sum_dim_int_list: "f32[8192, 4, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [-1], True)
        neg_default: "f32[8192, 4, 49, 49]" = torch.ops.aten.neg.default(div_1);  div_1 = None
        fma_default: "f32[8192, 4, 49, 49]" = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor);  neg_default = sum_dim_int_list = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        reshape_default_1: "f32[32768, 49, 49]" = torch.ops.aten.reshape.default(fma_default, _shape_param_1);  fma_default = _shape_param_1 = None
        return reshape_default_1

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
