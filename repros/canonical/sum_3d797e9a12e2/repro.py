"""
Standalone repro captured via capture_hook.
Label: timm_swin_base_patch4_window7_224_train
Pattern hash: 3d797e9a12e2
Shape hash: 092f66b3
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, div: "f32[8192, 4, 49, 49]", bmm_141: "f32[32768, 49, 49]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        expand_default: "f32[8192, 4, 49, 49]" = torch.ops.aten.expand.default(div, _shape_param_0);  _shape_param_0 = None
        reshape_default: "f32[32768, 49, 49]" = torch.ops.aten.reshape.default(expand_default, _shape_param_1);  expand_default = _shape_param_1 = None
        permute_default: "f32[32768, 49, 49]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1]);  reshape_default = None
        reshape_default_1: "f32[8192, 4, 49, 49]" = torch.ops.aten.reshape.default(bmm_141, _shape_param_2);  bmm_141 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        mul_tensor: "f32[8192, 4, 49, 49]" = torch.ops.aten.mul.Tensor(reshape_default_1, div);  reshape_default_1 = None
        sum_dim_int_list: "f32[8192, 4, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [-1], True)
        neg_default: "f32[8192, 4, 49, 49]" = torch.ops.aten.neg.default(div);  div = None
        fma_default: "f32[8192, 4, 49, 49]" = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor);  neg_default = sum_dim_int_list = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        reshape_default_2: "f32[32768, 49, 49]" = torch.ops.aten.reshape.default(fma_default, _shape_param_3);  fma_default = _shape_param_3 = None
        return (permute_default, reshape_default_2)


def _default_make_inputs():
    return [
    torch.randn(79691745, dtype=torch.float32, device='cuda').as_strided([8192, 4, 49, 49], [9728, 2432, 49, 1]),  # div
    torch.randn([32768, 49, 49], dtype=torch.float32, device='cuda'),
    [8192, 4, 49, 49],  # _shape_param_0
    [32768, 49, 49],  # _shape_param_1
    [8192, 4, 49, 49],  # _shape_param_2
    [32768, 49, 49],  # _shape_param_3
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
